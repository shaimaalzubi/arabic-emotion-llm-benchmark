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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
from collections import Counter

from openai import OpenAI


# ======================================================
# CONFIG — GPT-4o THREE-INDEPENDENT-RUN STABILITY ANALYSIS
# ======================================================
NUM_RUNS = 3
MODEL_NAME = "GPT-4o"
MODEL_ID = "gpt-4o"

VOTES = 1
TEMP_VOTES = 0.0

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_LABELS = ALLOWED_LABELS + [UNKNOWN_LABEL]
COVERED_LABELS = ALLOWED_LABELS
STRICT_UNKNOWN_SENTINEL = "__unknown__"

PROVIDER_NAME = "OpenAI"

MAX_WORKERS = 3
MAX_RETRIES = 6
BASE_BACKOFF = 0.8
JITTER = 0.40
REQUEST_SPACING = 0.0

INITIAL_RPS = 0.8
MIN_RPS = 0.2
MAX_RPS = 6.0

SEED = 42
random.seed(SEED)
DETERMINISTIC_JITTER = True

RUN_GROUP_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__GPT4o_SINGLE_x3"

PRINT_FIRST_N = 30
_print_lock = threading.Lock()


# ======================================================
# OUTPUT SCHEMA
# ======================================================
UNIFIED_PRED_COLS = [
    "row_id",
    "text",
    "gold_label",
    "prediction",
    "is_correct",
    "abstained",
    "unknown_reason",
    "calls_made",
    "raw_response",
    "provider",
    "model",
    "model_id",
    "run_group_id",
    "run_name",
    "run_index",
    "votes",
    "temp_votes",
    "max_workers",
]


# ======================================================
# UNKNOWN REASONS
# ======================================================
UNK_API_ERROR = "api_error"
UNK_EMPTY_MESSAGE = "empty_message_text"
UNK_INCOMPLETE = "incomplete_response"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"
UNK_OK = "ok"
UNK_FUTURE_EXCEPTION = "future_exception"


# ======================================================
# FLOAT-AWARE GLOBAL RATE LIMITER
# ======================================================
class AdaptiveRateLimiter:
    def __init__(self, initial_rps: float = 2.0, min_rps: float = 0.2, max_rps: float = 6.0):
        self._lock = threading.Lock()
        self._rps = float(initial_rps)
        self._min_rps = float(min_rps)
        self._max_rps = float(max_rps)
        self._cooldown_until = 0.0
        self._consecutive_429 = 0
        self._last_ts = 0.0

    def wait(self):
        while True:
            with self._lock:
                now = time.time()

                if now < self._cooldown_until:
                    sleep_for = self._cooldown_until - now
                else:
                    rps = max(self._min_rps, float(self._rps))
                    min_interval = 1.0 / rps
                    elapsed = now - self._last_ts

                    if self._last_ts == 0.0 or elapsed >= min_interval:
                        self._last_ts = now
                        return

                    sleep_for = min_interval - elapsed

            if sleep_for > 0:
                time.sleep(sleep_for)

    def note_429(self, retry_after_seconds: Optional[float] = None):
        with self._lock:
            self._consecutive_429 += 1
            factor = 0.75 if self._consecutive_429 == 1 else 0.6
            self._rps = max(self._min_rps, self._rps * factor)

            if retry_after_seconds is not None:
                extra = max(0.0, float(retry_after_seconds))
            else:
                extra = min(20.0, 1.5 * self._consecutive_429)

            self._cooldown_until = max(self._cooldown_until, time.time() + extra)

    def note_success(self):
        with self._lock:
            if self._consecutive_429 > 0:
                self._consecutive_429 = max(0, self._consecutive_429 - 1)
            self._rps = min(self._max_rps, self._rps + 0.15)


# ======================================================
# LOAD API
# ======================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")

client = OpenAI(api_key=api_key)


# ======================================================
# PATHS
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
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "GPT_runs")
PRED_DIR = os.path.join(OUTPUT_ROOT, "predictions")
FIG_DIR = os.path.join(OUTPUT_ROOT, "figures")
METRIC_DIR = os.path.join(OUTPUT_ROOT, "metrics")
DEBUG_DIR = os.path.join(OUTPUT_ROOT, "debug")
COMPARE_DIR = os.path.join(OUTPUT_ROOT, "comparisons")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8")


# ======================================================
# DATA HYGIENE
# ======================================================
required_cols = {"text", "emotion"}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"CSV is missing required columns: {sorted(missing)}")

df = df.dropna(subset=["text", "emotion"]).reset_index(drop=True)

df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
gold_map = {
    "angry": "anger",
    "mad": "anger",
    "happy": "joy",
    "hapy": "joy",
    "sad": "sadness"
}
df["emotion"] = df["emotion"].map(lambda x: gold_map.get(x, x))

bad_gold = sorted(set(df["emotion"]) - set(ALLOWED_LABELS))
if bad_gold:
    raise RuntimeError(
        f"Gold labels contain values not in ALLOWED_LABELS: {bad_gold}\nAllowed: {ALLOWED_LABELS}"
    )

df["row_id"] = range(len(df))

df["emotion"].value_counts().to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__DATASET_class_distribution.csv"),
    encoding="utf-8-sig",
)


# ======================================================
# MINIMAL ARABIC NORMALIZATION
# ======================================================
def normalize_arabic(text):
    text = str(text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(normalize_arabic)


# ======================================================
# PROMPT
# ======================================================
INSTRUCTIONS = (
    "You are an Arabic emotion classification engine. "
    "Classify the Arabic sentence into exactly one of these labels: "
    "joy, anger, sadness, fear, neutral. "
    "Return a strict JSON object with exactly one key: label. "
    "Do not return explanations or extra text."
)

PROMPT = """
Choose exactly ONE label from:
joy, anger, sadness, fear, neutral

Rules:
- Output must be valid JSON only.
- Use exactly this schema: {"label":"joy"}
- Lowercase only.
- If uncertain, choose the single best label from the five allowed labels.
- Do not output unknown.

Sentence: "{TEXT}"
""".strip()


# ======================================================
# STRUCTURED JSON SCHEMA
# ======================================================
def _text_config_for_model(model_id: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ALLOWED_LABELS
            }
        },
        "required": ["label"],
        "additionalProperties": False
    }

    return {
        "format": {
            "type": "json_schema",
            "name": "emotion_label",
            "schema": schema,
            "strict": True,
        }
    }


def _reasoning_config_for_model(model_id: str):
    # reasoning is only for reasoning-capable models (e.g. o3 / gpt-5)
    # GPT-4o does NOT support reasoning.effort
    if model_id.startswith("o") or model_id.startswith("gpt-5"):
        return {"effort": "low"}
    return None


# ======================================================
# PARSER
# ======================================================
def _strip_code_fences(text: str) -> str:
    t = "" if text is None else str(text)
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```", "", t)
    return t.strip()

def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    t = _strip_code_fences(text)
    m = re.search(r"\{.*?\}", t, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0).strip()

def _try_load_json_obj(blob: str):
    try:
        return json.loads(blob)
    except Exception:
        pass
    try:
        return json.loads(blob.replace("'", '"'))
    except Exception:
        return None

def _clean_label(lbl):
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

def parse_json_label_strict_with_reason(raw_text: str):
    raw = str(raw_text).strip() if raw_text is not None else ""
    if raw == "":
        return UNKNOWN_LABEL, UNK_EMPTY_MESSAGE

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            label_key = None
            for k in obj.keys():
                if str(k).strip().lower() == "label":
                    label_key = k
                    break
            if label_key is None:
                return UNKNOWN_LABEL, UNK_INVALID_JSON_MISSING_LABEL

            lab = _clean_label(obj.get(label_key))
            if lab == UNKNOWN_LABEL:
                return UNKNOWN_LABEL, UNK_INVALID_LABEL
            return lab, UNK_OK
    except Exception:
        pass

    blob = _extract_first_json_object(raw)
    if blob is None:
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

    lab = _clean_label(obj.get(label_key))
    if lab == UNKNOWN_LABEL:
        return UNKNOWN_LABEL, UNK_INVALID_LABEL

    return lab, UNK_OK


# ======================================================
# EXTRACT RESPONSE TEXT
# ======================================================
def extract_response_text(resp) -> str:
    try:
        t = (getattr(resp, "output_text", None) or "").strip()
        if t:
            return t
    except Exception:
        pass

    out = getattr(resp, "output", None)
    if isinstance(out, list):
        chunks = []
        for item in out:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue

            for c in content:
                ctype = getattr(c, "type", None)
                if ctype in ("output_text", "text"):
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt.strip())
                        continue

                for key in ("text", "value", "content"):
                    v = getattr(c, key, None)
                    if isinstance(v, str) and v.strip():
                        chunks.append(v.strip())
                        break

        joined = "\n".join(chunks).strip()
        if joined:
            return joined

    return ""


# ======================================================
# BACKOFF HELPERS
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

def _get_status_code(e: Exception) -> Optional[int]:
    sc = getattr(e, "status_code", None)
    if isinstance(sc, int):
        return sc
    resp = getattr(e, "response", None)
    if resp is not None:
        sc2 = getattr(resp, "status_code", None)
        if isinstance(sc2, int):
            return sc2
    return None

def _get_headers(e: Exception) -> dict:
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        return {}
    try:
        return dict(headers)
    except Exception:
        return {}

def _parse_seconds_value(v) -> Optional[float]:
    if v is None:
        return None
    try:
        s = str(v).strip()
        s = s.replace("s", "").strip()
        return float(s)
    except Exception:
        return None

def _get_retry_after_seconds(e: Exception) -> Optional[float]:
    h = _get_headers(e)
    ra = h.get("retry-after") or h.get("Retry-After")
    sec = _parse_seconds_value(ra)
    if sec is not None:
        return sec

    reset_req = h.get("x-ratelimit-reset-requests") or h.get("X-RateLimit-Reset-Requests")
    sec = _parse_seconds_value(reset_req)
    if sec is not None:
        return sec

    reset_tok = h.get("x-ratelimit-reset-tokens") or h.get("X-RateLimit-Reset-Tokens")
    sec = _parse_seconds_value(reset_tok)
    if sec is not None:
        return sec

    return None

def _is_transient_status(status_code: Optional[int]) -> bool:
    return status_code in (408, 409, 429, 500, 502, 503, 504, 529) or status_code is None


# ======================================================
# CALL OPENAI RESPONSES API
# ======================================================
def call_model(model_id, model_name, text, temperature, limiter: AdaptiveRateLimiter, sample_i=None):
    prompt = PROMPT.replace("{TEXT}", text)
    salt = f"{PROVIDER_NAME}|{model_name}|{sample_i}"
    last_err = None
    attempts_used = 0

    base_max_out = 256
    max_cap = 2048

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt

        try:
            limiter.wait()
            if REQUEST_SPACING > 0:
                time.sleep(REQUEST_SPACING)

            max_out = min(max_cap, int(base_max_out * (2 ** (attempt - 1))))

            kwargs = dict(
                model=model_id,
                input=prompt,
                instructions=INSTRUCTIONS,
                text=_text_config_for_model(model_id),
                max_output_tokens=max_out,
                temperature=float(temperature),
            )

            reasoning_cfg = _reasoning_config_for_model(model_id)
            if reasoning_cfg is not None:
                kwargs["reasoning"] = reasoning_cfg

            resp = client.responses.create(**kwargs)

            status = getattr(resp, "status", None)
            if status and status != "completed":
                inc = getattr(resp, "incomplete_details", None)
                reason = getattr(inc, "reason", None) if inc else None
                err_obj = getattr(resp, "error", None)
                last_err = (
                    f"{UNK_INCOMPLETE}__status={status}__reason={reason or 'unknown'}"
                    f"__error={str(err_obj)[:200]}"
                )

                raw_partial = extract_response_text(resp)
                if raw_partial.strip():
                    limiter.note_success()
                    return raw_partial, 0, None, attempts_used, status, str(inc), str(err_obj)

                _sleep_backoff(attempt, salt)
                continue

            limiter.note_success()

            raw = extract_response_text(resp)
            if not raw.strip():
                err_obj = getattr(resp, "error", None)
                inc = getattr(resp, "incomplete_details", None)
                last_err = (
                    f"{UNK_EMPTY_MESSAGE}__status={status}"
                    f"__incomplete={str(inc)[:160]}__error={str(err_obj)[:160]}"
                )
                _sleep_backoff(attempt, salt)
                continue

            return (
                raw,
                0,
                None,
                attempts_used,
                status,
                str(getattr(resp, "incomplete_details", None)),
                str(getattr(resp, "error", None)),
            )

        except Exception as e:
            status = _get_status_code(e)
            retry_after = _get_retry_after_seconds(e)
            msg = (str(e) or "")[:260]
            last_err = f"status_{status}__{msg}" if status is not None else (msg or "exception")

            if status == 429:
                limiter.note_429(retry_after_seconds=retry_after)

            if _is_transient_status(status):
                if status == 429 and retry_after is not None and retry_after > 0:
                    time.sleep(min(60.0, retry_after))
                _sleep_backoff(attempt, salt)
                continue

            return "", 1, last_err, attempts_used, None, None, None

    return "", 1, (last_err or "max_retries_exceeded"), attempts_used, None, None, None


# ======================================================
# SINGLE INFERENCE
# ======================================================
def classify_single(model_id, model_name, text, limiter: AdaptiveRateLimiter, sample_i=None):
    raw, err_flag, err_text, calls_made, resp_status, inc_details, err_obj = call_model(
        model_id=model_id,
        model_name=model_name,
        text=text,
        temperature=TEMP_VOTES,
        limiter=limiter,
        sample_i=sample_i
    )

    if err_flag == 1:
        return UNKNOWN_LABEL, 1, 1, int(calls_made), "", (err_text or UNK_API_ERROR)

    pred, reason = parse_json_label_strict_with_reason(raw)
    abstained = 1 if pred == UNKNOWN_LABEL else 0

    return pred, 0, abstained, int(calls_made), raw, reason


# ======================================================
# PLOT CONFUSION MATRIX
# ======================================================
def save_confusion_matrix(cm, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ======================================================
# PREFLIGHT MODEL TEST
# ======================================================
def test_model():
    print("\n[TEST] Testing GPT-4o with 1 request...")
    limiter = AdaptiveRateLimiter(initial_rps=0.5, min_rps=0.2, max_rps=2.0)
    test_text = "اليوم زعلانه لان فيه غيوم وما نزل مطر"

    raw, err_flag, err_text, calls, status, incomplete_details, err_obj = call_model(
        MODEL_ID, MODEL_NAME, test_text, temperature=TEMP_VOTES, limiter=limiter, sample_i="TEST"
    )

    parsed, parse_reason = parse_json_label_strict_with_reason(raw) if raw else (UNKNOWN_LABEL, "no_raw")

    print("-----")
    print(f"Model: {MODEL_NAME} / {MODEL_ID}")
    print(f"err_flag: {err_flag} | calls: {calls}")
    print(f"status: {status}")
    print(f"err_text: {err_text}")
    print(f"incomplete_details: {incomplete_details}")
    print(f"error_obj: {err_obj}")
    print(f"raw_head: {(raw[:200] if raw else '<<EMPTY>>')}")
    print(f"parsed: {parsed} | parse_reason: {parse_reason}")
    print("-----")

    pd.DataFrame([{
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "err_flag": err_flag,
        "calls": calls,
        "status": status,
        "err_text": err_text,
        "incomplete_details": incomplete_details,
        "error_obj": err_obj,
        "raw": raw,
        "parsed": parsed,
        "parse_reason": parse_reason,
    }]).to_csv(
        os.path.join(DEBUG_DIR, f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__PREFLIGHT_DEBUG.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# ======================================================
# RUN ONE FULL BENCHMARK
# ======================================================
def run_one_benchmark(run_index: int):
    run_name = f"Run{run_index}"
    run_id = f"{RUN_GROUP_ID}__{run_name}"

    print(f"\nRunning {MODEL_NAME} — {run_name} (VOTES={VOTES}, TEMP={TEMP_VOTES})...")

    LIMITER = AdaptiveRateLimiter(
        initial_rps=INITIAL_RPS,
        min_rps=MIN_RPS,
        max_rps=MAX_RPS,
    )

    n = len(df)
    preds = [None] * n
    abstained_arr = [0] * n
    unknown_reasons = [None] * n
    calls_made_arr = [0] * n
    raw_response_arr = [None] * n

    err_calls_total = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                classify_single,
                MODEL_ID,
                MODEL_NAME,
                df["text"].iloc[i],
                LIMITER,
                f"{run_name}_row{i}"
            ): i
            for i in range(n)
        }

        for fut in tqdm(as_completed(futures), total=n, desc=f"{MODEL_NAME}-{run_name}"):
            i = futures[fut]

            try:
                pred, err_calls, abst_inc, calls_made, raw, reason = fut.result()
            except Exception as e:
                pred, err_calls, abst_inc, calls_made, raw, reason = (
                    UNKNOWN_LABEL, 1, 1, 0, "", f"{UNK_FUTURE_EXCEPTION}__{str(e)[:200]}"
                )

            preds[i] = pred
            abstained_arr[i] = abst_inc
            unknown_reasons[i] = reason
            calls_made_arr[i] = int(calls_made)
            raw_response_arr[i] = raw

            err_calls_total += err_calls

            if i < PRINT_FIRST_N:
                with _print_lock:
                    print("-----")
                    print(
                        f"[{i+1}/{n}] {run_name}  GOLD={df['emotion'].iloc[i]}  "
                        f"PRED={pred}  abst={abst_inc}  reason={reason}  calls={calls_made}"
                    )
                    print(f"TEXT: {df['text'].iloc[i]}")
                    print(f"RAW : {raw if raw else '<<EMPTY>>'}")
                    print("-----\n")

    y_true = df["emotion"].tolist()
    y_pred = preds

    unknown_total = y_pred.count(UNKNOWN_LABEL)
    unknown_rate = unknown_total / len(y_pred)
    coverage = 1 - unknown_rate

    y_pred_strict = [p if p != UNKNOWN_LABEL else STRICT_UNKNOWN_SENTINEL for p in y_pred]
    strict_accuracy = accuracy_score(y_true, y_pred_strict)
    strict_macro_f1 = f1_score(y_true, y_pred_strict, average="macro", labels=ALLOWED_LABELS, zero_division=0)
    strict_micro_f1 = f1_score(y_true, y_pred_strict, average="micro", labels=ALLOWED_LABELS, zero_division=0)
    strict_weighted_f1 = f1_score(y_true, y_pred_strict, average="weighted", labels=ALLOWED_LABELS, zero_division=0)

    report_strict_5 = classification_report(
        y_true, y_pred_strict, labels=ALLOWED_LABELS, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_strict_5).transpose().to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__{run_id}__{MODEL_NAME}__REPORT_STRICT_5LABELS.csv"),
        encoding="utf-8-sig",
    )

    cm_with_unknown = confusion_matrix(y_true, y_pred, labels=STRICT_LABELS)
    save_confusion_matrix(
        cm_with_unknown,
        STRICT_LABELS,
        title=f"{MODEL_NAME} Confusion Matrix (incl. unknown) — {run_name}",
        outpath=os.path.join(FIG_DIR, f"{PROVIDER_NAME}__{run_id}__{MODEL_NAME}__CM_WITH_UNKNOWN.png")
    )

    cm_5 = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
    save_confusion_matrix(
        cm_5,
        ALLOWED_LABELS,
        title=f"{MODEL_NAME} Confusion Matrix (5 labels) — {run_name}",
        outpath=os.path.join(FIG_DIR, f"{PROVIDER_NAME}__{run_id}__{MODEL_NAME}__CM_5LABELS.png")
    )

    mask = [p != UNKNOWN_LABEL for p in y_pred]
    y_true_cov = [yt for yt, m in zip(y_true, mask) if m]
    y_pred_cov = [yp for yp in y_pred if yp != UNKNOWN_LABEL]

    if len(y_true_cov) > 0:
        covered_accuracy = accuracy_score(y_true_cov, y_pred_cov)
        covered_macro_f1 = f1_score(y_true_cov, y_pred_cov, average="macro", labels=COVERED_LABELS, zero_division=0)
        covered_micro_f1 = f1_score(y_true_cov, y_pred_cov, average="micro", labels=COVERED_LABELS, zero_division=0)
        covered_weighted_f1 = f1_score(y_true_cov, y_pred_cov, average="weighted", labels=COVERED_LABELS, zero_division=0)
    else:
        covered_accuracy = covered_macro_f1 = covered_micro_f1 = covered_weighted_f1 = 0.0

    out_pred_df = pd.DataFrame({
        "row_id": df["row_id"],
        "text": df["text"],
        "gold_label": df["emotion"],
        "prediction": y_pred,
        "is_correct": [int(g == p) for g, p in zip(y_true, y_pred)],
        "abstained": abstained_arr,
        "unknown_reason": unknown_reasons,
        "calls_made": calls_made_arr,
        "raw_response": raw_response_arr,
        "provider": PROVIDER_NAME,
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "run_group_id": RUN_GROUP_ID,
        "run_name": run_name,
        "run_index": run_index,
        "votes": VOTES,
        "temp_votes": TEMP_VOTES,
        "max_workers": MAX_WORKERS,
    })

    out_pred_df = out_pred_df[UNIFIED_PRED_COLS]
    pred_path = os.path.join(PRED_DIR, f"{PROVIDER_NAME}__{run_id}__{MODEL_NAME}__PREDICTIONS.csv")
    out_pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    reason_counter = Counter(unknown_reasons)
    metrics_row = {
        "provider": PROVIDER_NAME,
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "run_group_id": RUN_GROUP_ID,
        "run_name": run_name,
        "run_index": run_index,
        "setup": "zero-shot + single inference repeated independently three times for GPT-4o stability analysis",
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
        "abstention_rate": sum(abstained_arr) / len(df),
        "error_calls_per_sample": err_calls_total / len(df),

        "covered_accuracy": covered_accuracy,
        "covered_macro_f1": covered_macro_f1,
        "covered_micro_f1": covered_micro_f1,
        "covered_weighted_f1": covered_weighted_f1,

        "unknown_reason_api_error": int(reason_counter.get(UNK_API_ERROR, 0)),
        "unknown_reason_invalid_json_no_object": int(reason_counter.get(UNK_INVALID_JSON_NO_OBJECT, 0)),
        "unknown_reason_invalid_json_parse_error": int(reason_counter.get(UNK_INVALID_JSON_PARSE_ERROR, 0)),
        "unknown_reason_invalid_json_missing_label": int(reason_counter.get(UNK_INVALID_JSON_MISSING_LABEL, 0)),
        "unknown_reason_invalid_label": int(reason_counter.get(UNK_INVALID_LABEL, 0)),
        "unknown_reason_empty_message": int(reason_counter.get(UNK_EMPTY_MESSAGE, 0)),
        "unknown_reason_incomplete_response": sum(
            1 for r in unknown_reasons if isinstance(r, str) and r.startswith(UNK_INCOMPLETE)
        ),
    }

    return out_pred_df, metrics_row


# ======================================================
# COMPARE RUNS
# ======================================================
def compare_runs(run_pred_dfs, metrics_rows):
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(
        METRIC_DIR,
        f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__{MODEL_NAME}__3RUNS_SUMMARY.csv"
    )
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    r1 = run_pred_dfs[0][["row_id", "text", "gold_label", "prediction", "is_correct"]].rename(
        columns={"prediction": "pred_run1", "is_correct": "correct_run1"}
    )
    r2 = run_pred_dfs[1][["row_id", "prediction", "is_correct"]].rename(
        columns={"prediction": "pred_run2", "is_correct": "correct_run2"}
    )
    r3 = run_pred_dfs[2][["row_id", "prediction", "is_correct"]].rename(
        columns={"prediction": "pred_run3", "is_correct": "correct_run3"}
    )

    merged = r1.merge(r2, on="row_id").merge(r3, on="row_id")

    merged["same_prediction_all_3"] = (
        (merged["pred_run1"] == merged["pred_run2"]) &
        (merged["pred_run1"] == merged["pred_run3"])
    ).astype(int)

    merged["same_correctness_all_3"] = (
        (merged["correct_run1"] == merged["correct_run2"]) &
        (merged["correct_run1"] == merged["correct_run3"])
    ).astype(int)

    merged["all_three_correct"] = (
        (merged["correct_run1"] == 1) &
        (merged["correct_run2"] == 1) &
        (merged["correct_run3"] == 1)
    ).astype(int)

    merged["all_three_wrong"] = (
        (merged["correct_run1"] == 0) &
        (merged["correct_run2"] == 0) &
        (merged["correct_run3"] == 0)
    ).astype(int)

    merged["changed_prediction_between_runs"] = (merged["same_prediction_all_3"] == 0).astype(int)

    merged["wrong_in_at_least_one_run"] = (
        (merged["correct_run1"] == 0) |
        (merged["correct_run2"] == 0) |
        (merged["correct_run3"] == 0)
    ).astype(int)

    merged["error_pattern"] = (
        merged["correct_run1"].astype(str) +
        merged["correct_run2"].astype(str) +
        merged["correct_run3"].astype(str)
    )

    merged["prediction_pattern"] = (
        merged["pred_run1"].astype(str) + " | " +
        merged["pred_run2"].astype(str) + " | " +
        merged["pred_run3"].astype(str)
    )

    full_compare_path = os.path.join(
        COMPARE_DIR,
        f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__{MODEL_NAME}__ROW_LEVEL_COMPARISON.csv"
    )
    merged.to_csv(full_compare_path, index=False, encoding="utf-8-sig")

    changed_only = merged[merged["changed_prediction_between_runs"] == 1].copy()
    changed_only.to_csv(
        os.path.join(
            COMPARE_DIR,
            f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__{MODEL_NAME}__ROWS_CHANGED_ACROSS_RUNS.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    all_wrong = merged[merged["all_three_wrong"] == 1].copy()
    all_wrong.to_csv(
        os.path.join(
            COMPARE_DIR,
            f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__{MODEL_NAME}__ROWS_WRONG_IN_ALL_3_RUNS.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    unstable_correctness = merged[merged["same_correctness_all_3"] == 0].copy()
    unstable_correctness.to_csv(
        os.path.join(
            COMPARE_DIR,
            f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__{MODEL_NAME}__ROWS_CORRECTNESS_CHANGED.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    pairwise_summary = pd.DataFrame([
        {
            "pair": "Run1_vs_Run2",
            "same_prediction_count": int((merged["pred_run1"] == merged["pred_run2"]).sum()),
            "different_prediction_count": int((merged["pred_run1"] != merged["pred_run2"]).sum()),
            "same_correctness_count": int((merged["correct_run1"] == merged["correct_run2"]).sum()),
            "different_correctness_count": int((merged["correct_run1"] != merged["correct_run2"]).sum()),
        },
        {
            "pair": "Run1_vs_Run3",
            "same_prediction_count": int((merged["pred_run1"] == merged["pred_run3"]).sum()),
            "different_prediction_count": int((merged["pred_run1"] != merged["pred_run3"]).sum()),
            "same_correctness_count": int((merged["correct_run1"] == merged["correct_run3"]).sum()),
            "different_correctness_count": int((merged["correct_run1"] != merged["correct_run3"]).sum()),
        },
        {
            "pair": "Run2_vs_Run3",
            "same_prediction_count": int((merged["pred_run2"] == merged["pred_run3"]).sum()),
            "different_prediction_count": int((merged["pred_run2"] != merged["pred_run3"]).sum()),
            "same_correctness_count": int((merged["correct_run2"] == merged["correct_run3"]).sum()),
            "different_correctness_count": int((merged["correct_run2"] != merged["correct_run3"]).sum()),
        },
    ])

    pairwise_summary.to_csv(
        os.path.join(
            COMPARE_DIR,
            f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__{MODEL_NAME}__PAIRWISE_SUMMARY.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    global_summary = pd.DataFrame([{
        "run_group_id": RUN_GROUP_ID,
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "total_rows": len(merged),
        "same_prediction_all_3_count": int(merged["same_prediction_all_3"].sum()),
        "changed_prediction_across_runs_count": int(merged["changed_prediction_between_runs"].sum()),
        "all_three_correct_count": int(merged["all_three_correct"].sum()),
        "all_three_wrong_count": int(merged["all_three_wrong"].sum()),
        "correctness_changed_count": int((merged["same_correctness_all_3"] == 0).sum()),
        "wrong_in_at_least_one_run_count": int(merged["wrong_in_at_least_one_run"].sum()),
    }])

    global_summary.to_csv(
        os.path.join(
            COMPARE_DIR,
            f"{PROVIDER_NAME}__GROUP{RUN_GROUP_ID}__{MODEL_NAME}__GLOBAL_COMPARISON_SUMMARY.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 3-RUN COMPARISON SUMMARY =====")
    print(global_summary.to_string(index=False))
    print("\nPer-run metrics:")
    print(metrics_df[[
        "run_name",
        "strict_accuracy_unknown_wrong",
        "strict_macro_f1",
        "strict_micro_f1",
        "strict_weighted_f1",
        "coverage",
        "unknown_rate",
        "abstention_rate"
    ]].to_string(index=False))
    print("\nPairwise summary:")
    print(pairwise_summary.to_string(index=False))


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    test_model()

    all_run_pred_dfs = []
    all_metrics_rows = []

    for run_idx in range(1, NUM_RUNS + 1):
        run_pred_df, metrics_row = run_one_benchmark(run_idx)
        all_run_pred_dfs.append(run_pred_df)
        all_metrics_rows.append(metrics_row)

    compare_runs(all_run_pred_dfs, all_metrics_rows)

    print("\nGPT-4o three-independent-run stability benchmark completed successfully.")
    print("RUN_GROUP_ID:", RUN_GROUP_ID)
    print("Outputs saved under:")
    print(" - predictions :", PRED_DIR)
    print(" - figures     :", FIG_DIR)
    print(" - metrics     :", METRIC_DIR)
    print(" - debug       :", DEBUG_DIR)
    print(" - comparisons :", COMPARE_DIR)