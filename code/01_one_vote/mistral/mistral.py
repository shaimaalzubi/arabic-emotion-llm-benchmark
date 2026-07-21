import os
import re
import time
import json
import random
import hashlib
import threading
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

from mistralai import Mistral  # pip install mistralai


# ===============================
# CONFIG
# ===============================
VOTES = 1
TEMP_VOTES = 0.0

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_UNKNOWN_SENTINEL = "unknown"

PROVIDER_NAME = "Mistral"

# Mistral models evaluated in the manuscript.
# Keep the exact API aliases used for the reported benchmark.
MODELS = {
    "Mistral-Medium": "mistral-medium-latest",
    "Mistral-Large": "mistral-large-latest",
}

# ===============================
# SPEED / RELIABILITY
# ===============================
MAX_WORKERS = 8
MAX_RETRIES = 6
BASE_BACKOFF = 0.8
JITTER = 0.25

TARGET_RPM = 240
MIN_INTERVAL_SEC = max(0.02, 60.0 / max(1, TARGET_RPM))

MAX_TOKENS = 16

SEED = 42
random.seed(SEED)
DETERMINISTIC_JITTER = True

PRINT_FIRST_N = 40
STORE_VOTES_LIST = True

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__MISTRAL_V1"


# ===============================
# OUTPUT SCHEMA
# ===============================
UNIFIED_PRED_COLS = [
    "text", "gold_label", "prediction",
    "abstained", "vote_pattern", "unknown_votes", "unknown_reason", "calls_made",
    "provider", "model", "run_id",
    "votes", "temp_votes", "max_workers",
]

# ===============================
# UNKNOWN REASONS
# ===============================
UNK_API_ERROR = "api_error"
UNK_TIMEOUT = "timeout"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"
UNK_OK = "ok"
UNK_REGEX_LABEL = "regex_label"


# ======================================================
# Minimal Arabic Normalization
# ======================================================
def normalize_arabic(text):
    text = str(text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ======================================================
# Parser helpers
# ======================================================
def _strip_code_fences(text: str) -> str:
    t = "" if text is None else str(text)
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```", "", t)
    return t.strip()


def clean_label(lbl):
    if lbl is None:
        return UNKNOWN_LABEL
    s = str(lbl).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return UNKNOWN_LABEL

    first = s.split(" ")[0]
    if first in ALLOWED_LABELS:
        return first

    for lab in ALLOWED_LABELS:
        if re.search(rf"\b{lab}\b", s):
            return lab

    return UNKNOWN_LABEL


def _extract_first_json_object_balanced(text: str) -> Optional[str]:
    if not text:
        return None
    t = _strip_code_fences(text)

    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1].strip()

    return None


def _try_load_json_obj(blob: str):
    try:
        return json.loads(blob)
    except Exception:
        pass
    try:
        repaired = blob.replace("'", '"')
        return json.loads(repaired)
    except Exception:
        return None


def _explicit_label_regex_fallback(raw: str) -> Optional[str]:
    if not raw:
        return None
    low = _strip_code_fences(raw).lower()
    m = re.search(
        r"""\blabel\b\s*[:=]\s*["']?\s*(joy|anger|sadness|fear|neutral)\s*["']?""",
        low
    )
    return m.group(1) if m else None


def parse_json_label_strict_with_reason(raw_text: str):
    raw = str(raw_text).strip() if raw_text is not None else ""
    if raw == "":
        return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT

    blob = _extract_first_json_object_balanced(raw)
    if blob is None:
        lab = _explicit_label_regex_fallback(raw)
        if lab is not None:
            return lab, UNK_REGEX_LABEL
        return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT

    obj = _try_load_json_obj(blob)
    if obj is None:
        return UNKNOWN_LABEL, UNK_INVALID_JSON_PARSE_ERROR
    if not isinstance(obj, dict):
        return UNKNOWN_LABEL, UNK_INVALID_JSON_MISSING_LABEL

    label_key = None
    for k in obj.keys():
        if str(k).strip().lower() == "label":
            label_key = k
            break
    if label_key is None:
        return UNKNOWN_LABEL, UNK_INVALID_JSON_MISSING_LABEL

    lab = clean_label(obj.get(label_key))
    if lab == UNKNOWN_LABEL:
        return UNKNOWN_LABEL, UNK_INVALID_LABEL

    return lab, UNK_OK


# ======================================================
# Backoff helpers
# ======================================================
def _det_jitter(seed: int, attempt: int, salt: str) -> float:
    key = f"{seed}|{attempt}|{salt}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    val = int(h[:8], 16) / float(16**8)
    return val * JITTER


def _sleep_backoff(attempt: int, salt: str):
    base = BASE_BACKOFF * (2 ** (attempt - 1))
    jitter = _det_jitter(SEED, attempt, salt) if DETERMINISTIC_JITTER else random.uniform(0, JITTER)
    time.sleep(base + jitter)


# ======================================================
# Simple global rate limiter (thread-safe)
# ======================================================
class GlobalRateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval = float(min_interval_sec)
        self.lock = threading.Lock()
        self.last_ts = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                if self.last_ts == 0.0:
                    self.last_ts = now
                    return
                elapsed = now - self.last_ts
                if elapsed >= self.min_interval:
                    self.last_ts = now
                    return
                sleep_for = self.min_interval - elapsed
            time.sleep(max(0.0, sleep_for))


# ======================================================
# 1) Load API Key
# ======================================================
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("ERROR: MISTRAL_API_KEY not found in .env")

client = Mistral(api_key=API_KEY)


# ======================================================
# 2) Paths + Data
# ======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "EmotionsFile_GoldLabelCSV.csv",
)
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "one_vote")
PRED_DIR = os.path.join(OUTPUT_ROOT, "predictions")
METRIC_DIR = os.path.join(OUTPUT_ROOT, "metrics")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8")

required_cols = {"text", "emotion"}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"CSV is missing required columns: {sorted(missing)}")

df = df.dropna(subset=["text", "emotion"]).reset_index(drop=True)
df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

gold_map = {"angry": "anger", "mad": "anger", "happy": "joy", "hapy": "joy", "sad": "sadness"}
df["emotion"] = df["emotion"].map(lambda x: gold_map.get(x, x))

bad_gold = sorted(set(df["emotion"]) - set(ALLOWED_LABELS))
if bad_gold:
    raise RuntimeError(f"Gold labels contain values not in ALLOWED_LABELS: {bad_gold}\nAllowed: {ALLOWED_LABELS}")

df["text"] = df["text"].apply(normalize_arabic)

df["emotion"].value_counts().to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__DATASET_class_distribution.csv"),
    encoding="utf-8-sig"
)


# ======================================================
# 3) Strict JSON Prompt
# ======================================================
SYSTEM_MSG = (
    "You are an Arabic emotion classifier. "
    "Return ONLY a valid JSON object. "
    "No explanation, no markdown, no extra text. "
    "The JSON must be exactly one object with one key: label. "
    "Allowed labels: joy, anger, sadness, fear, neutral. "
    "Lowercase only. "
    "If uncertain, return neutral."
)

PROMPT_TEMPLATE = (
    "Classify the Arabic sentence into exactly one emotion label.\n"
    "Return only JSON in this exact schema: {\"label\":\"neutral\"}\n"
    "Sentence: \"{TEXT}\""
)


# ======================================================
# 4) Mistral call (robust + fast)
# ======================================================
limiter = GlobalRateLimiter(MIN_INTERVAL_SEC)


def _extract_mistral_text(resp) -> str:
    try:
        ch0 = resp.choices[0]
        msg = getattr(ch0, "message", None)
        if msg is None and isinstance(ch0, dict):
            msg = ch0.get("message")

        content = getattr(msg, "content", None) if msg is not None else None
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict):
                    if "text" in blk:
                        parts.append(str(blk["text"]))
                    elif "content" in blk:
                        parts.append(str(blk["content"]))
                    else:
                        parts.append(str(blk))
                else:
                    txt = getattr(blk, "text", None)
                    if txt is not None:
                        parts.append(str(txt))
                    else:
                        parts.append(str(blk))
            return "".join(parts).strip()

        if content is not None:
            return str(content).strip()

    except Exception:
        pass

    try:
        return json.dumps(resp, ensure_ascii=False)
    except Exception:
        return str(resp)


def call_model(text: str, model_id: str, temperature: float, sample_i=None):
    prompt = PROMPT_TEMPLATE.replace("{TEXT}", text)
    salt = f"{PROVIDER_NAME}|{model_id}|{sample_i}"
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.wait()

            resp = client.chat.complete(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(temperature),
                max_tokens=MAX_TOKENS,
                random_seed=SEED,
                response_format={"type": "json_object"},
                safe_prompt=False,
                stream=False,
            )

            raw_text = _extract_mistral_text(resp).strip()

            blob = _extract_first_json_object_balanced(raw_text)
            if blob is None:
                last_err = f"{UNK_INVALID_JSON_NO_OBJECT}__raw_head={raw_text[:160]}"
                _sleep_backoff(attempt, salt)
                continue

            return blob.strip(), 0, None, attempt

        except Exception as e:
            msg = repr(e)[:800]
            if "timeout" in msg.lower():
                last_err = f"{UNK_TIMEOUT}__{msg}"
            else:
                last_err = f"{UNK_API_ERROR}__{msg}"
            _sleep_backoff(attempt, salt)
            continue

    return "", 1, (last_err or "max_retries_exceeded"), MAX_RETRIES


# ======================================================
# 5) Run Benchmark per model
# ======================================================
summary_rows = []

for MODEL_NAME, MODEL_ID in MODELS.items():
    print(f"\nRunning {MODEL_NAME} (VOTES=1, TEMP={TEMP_VOTES})...")
    print(
        f"Runtime config: workers={MAX_WORKERS}, target_rpm={TARGET_RPM}, "
        f"min_interval={MIN_INTERVAL_SEC:.3f}s, max_tokens={MAX_TOKENS}"
    )

    test_text = "انا مبسوطة اليوم"
    raw_t, err_t, err_msg_t, att_t = call_model(test_text, MODEL_ID, TEMP_VOTES, sample_i="TEST")
    print("[TEST] err_flag:", err_t, "| attempts:", att_t)
    if err_t == 1:
        print("[TEST] err:", err_msg_t)
        raise RuntimeError("Test call failed — fix Mistral API/model/quota before full benchmark.")
    else:
        print("[TEST] raw:", raw_t)

    printed_next = [0]
    pending_print = {}
    print_lock = threading.Lock()

    def queue_print(i: int, gold: str, row: dict, text: str):
        if i >= PRINT_FIRST_N:
            return
        line = (
            f"-----\n"
            f"[{i:04d}] GOLD={gold:<7}  PRED={row['pred']:<7}  "
            f"abst={row['abstained']}  reason={row['unknown_reason']}  calls={row['calls_made']}\n"
            f"TEXT: {text}\n"
            f"RAW : {row['raw']}\n"
        )
        with print_lock:
            pending_print[i] = line
            while printed_next[0] in pending_print:
                print(pending_print.pop(printed_next[0]), flush=True)
                printed_next[0] += 1

    def classify_one(i: int):
        text = df.loc[i, "text"]
        gold = df.loc[i, "emotion"]

        raw, err_flag, err_text, attempts_used = call_model(text, MODEL_ID, TEMP_VOTES, sample_i=i)
        calls_made = int(attempts_used)

        if err_flag == 1:
            row = {
                "pred": UNKNOWN_LABEL,
                "err_calls": int(attempts_used),
                "abstained": 1,
                "calls_made": calls_made,
                "vote_pattern": "0",
                "unknown_votes": 1,
                "unknown_reason": UNK_API_ERROR,
                "raw": f"__API_ERROR__ {str(err_text)[:800]}",
            }
            return i, gold, text, row

        pred, parse_reason = parse_json_label_strict_with_reason(raw)
        if pred == UNKNOWN_LABEL:
            row = {
                "pred": pred,
                "err_calls": 0,
                "abstained": 1,
                "calls_made": calls_made,
                "vote_pattern": "0",
                "unknown_votes": 1,
                "unknown_reason": parse_reason,
                "raw": raw[:800],
            }
            return i, gold, text, row

        row = {
            "pred": pred,
            "err_calls": 0,
            "abstained": 0,
            "calls_made": calls_made,
            "vote_pattern": "1",
            "unknown_votes": 0,
            "unknown_reason": UNK_OK,
            "raw": raw[:800],
        }
        return i, gold, text, row

    n = len(df)
    preds = [None] * n
    abstained_arr = [0] * n
    vote_patterns = [None] * n
    unknown_votes_arr = [0] * n
    unknown_reasons = [None] * n
    calls_made_arr = [0] * n
    votes_list_arr = [None] * n
    raw_preview_arr = [None] * n

    error_calls_total = 0
    abstain_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_one, i): i for i in range(n)}

        for fut in tqdm(as_completed(futures), total=n, desc=MODEL_NAME):
            i, gold, text, row = fut.result()

            preds[i] = row["pred"]
            abstained_arr[i] = int(row["abstained"])
            vote_patterns[i] = row["vote_pattern"]
            unknown_votes_arr[i] = int(row["unknown_votes"])
            unknown_reasons[i] = row["unknown_reason"]
            calls_made_arr[i] = int(row["calls_made"])
            raw_preview_arr[i] = row["raw"]

            if STORE_VOTES_LIST:
                votes_list_arr[i] = json.dumps([row["pred"]], ensure_ascii=False)

            error_calls_total += int(row["err_calls"])
            abstain_count += int(row["abstained"])

            queue_print(i, gold, row, text)

    y_true = df["emotion"].tolist()
    y_pred = preds

    unknown_total = y_pred.count(UNKNOWN_LABEL)
    unknown_rate = unknown_total / len(y_pred)
    coverage = 1 - unknown_rate

    pd.Series(unknown_reasons).value_counts(dropna=False).to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__UNKNOWN_REASON_DISTRIBUTION.csv"),
        encoding="utf-8-sig"
    )

    y_pred_strict = [p if p != UNKNOWN_LABEL else STRICT_UNKNOWN_SENTINEL for p in y_pred]

    strict_accuracy = accuracy_score(y_true, y_pred_strict)
    strict_macro_f1 = f1_score(y_true, y_pred_strict, average="macro", labels=ALLOWED_LABELS, zero_division=0)
    strict_micro_f1 = f1_score(y_true, y_pred_strict, average="micro", labels=ALLOWED_LABELS, zero_division=0)
    strict_weighted_f1 = f1_score(y_true, y_pred_strict, average="weighted", labels=ALLOWED_LABELS, zero_division=0)

    out_df = pd.DataFrame({
        "text": df["text"],
        "gold_label": df["emotion"],
        "prediction": y_pred,
        "abstained": abstained_arr,
        "vote_pattern": vote_patterns,
        "unknown_votes": unknown_votes_arr,
        "unknown_reason": unknown_reasons,
        "calls_made": calls_made_arr,
        "provider": PROVIDER_NAME,
        "model": MODEL_NAME,
        "run_id": RUN_ID,
        "votes": VOTES,
        "temp_votes": TEMP_VOTES,
        "max_workers": MAX_WORKERS,
    })

    out_df["raw_preview"] = raw_preview_arr

    if STORE_VOTES_LIST:
        out_df["votes_list"] = votes_list_arr
        cols = UNIFIED_PRED_COLS + ["votes_list", "raw_preview"]
    else:
        cols = UNIFIED_PRED_COLS + ["raw_preview"]

    out_df = out_df[cols]
    out_df.to_csv(
        os.path.join(PRED_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__PREDICTIONS.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary_rows.append({
        "provider": PROVIDER_NAME,
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "setup": "zero-shot + single inference (V1) — Mistral SDK chat.complete + JSON mode",
        "votes": VOTES,
        "temp_votes": TEMP_VOTES,
        "max_workers": MAX_WORKERS,
        "total_samples": len(df),
        "strict_accuracy_unknown_wrong": strict_accuracy,
        "strict_macro_f1": strict_macro_f1,
        "strict_micro_f1": strict_micro_f1,
        "strict_weighted_f1": strict_weighted_f1,
        "coverage": coverage,
        "unknown_rate": unknown_rate,
        "unknown_total": int(unknown_total),
        "abstention_rate": abstain_count / len(df),
        "error_calls_per_sample": error_calls_total / len(df),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_FAST_JSON.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nMistral benchmark completed successfully for all manuscript models.")
print("RUN_ID:", RUN_ID)
print("Models run:", list(MODELS.keys()))
print("Outputs saved under:")
print(" - predictions:", PRED_DIR)
print(" - metrics    :", METRIC_DIR)