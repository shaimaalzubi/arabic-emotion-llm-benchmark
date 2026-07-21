import os
import re
import time
import json
import random
import hashlib
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI


# ===============================
# CONFIG
# ===============================
VOTES = 1
TEMP_VOTES = 0.0

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_LABELS = ALLOWED_LABELS + [UNKNOWN_LABEL]
COVERED_LABELS = ALLOWED_LABELS
STRICT_UNKNOWN_SENTINEL = "__unknown__"

PROVIDER_NAME = "OpenAI"

MODELS = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "GPT-5": "gpt-5",
    "OpenAI-o3": "o3",
}

# ===============================
# SPEED SETTINGS
# ===============================
REASONING_MODELS = {"o3"}
NO_TEMPERATURE_MODELS = {"gpt-5", "o3"}

MAX_WORKERS = 6

MAX_RETRIES_BY_MODEL = {
    "gpt-4o": 3,
    "gpt-4.1-mini": 3,
    "gpt-5": 2,
    "o3": 2,
}

BASE_BACKOFF = 0.35
JITTER = 0.15
REQUEST_SPACING = 0.0

INITIAL_RPS = 2.5
MIN_RPS = 0.5
MAX_RPS = 8.0

MAX_OUTPUT_TOKENS_BY_MODEL = {
    "gpt-4o": 24,
    "gpt-4.1-mini": 24,
    "gpt-5": 400,
    "o3": 256,
}

SEED = 42
random.seed(SEED)
DETERMINISTIC_JITTER = True

STORE_VOTES_LIST = True
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__FEWSHOT_SINGLE_FAST"

PRINT_FIRST_N = 20
_print_lock = threading.Lock()


# ===============================
# OUTPUT SCHEMA
# ===============================
UNIFIED_PRED_COLS = [
    "text", "gold_label", "prediction",
    "abstained", "vote_pattern", "unknown_votes", "unknown_reason", "calls_made",
    "provider", "model", "model_id", "run_id",
    "votes", "temp_votes", "max_workers",
]
if STORE_VOTES_LIST:
    UNIFIED_PRED_COLS.append("votes_list")


# ===============================
# UNKNOWN REASONS
# ===============================
UNK_API_ERROR = "api_error"
UNK_EMPTY_MESSAGE = "empty_message_text"
UNK_INCOMPLETE = "incomplete_response"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"
UNK_ALL_FAILED = "all_failed"
UNK_ALL_DIFFERENT_111 = "all_different_111"
UNK_NO_MAJORITY_11U = "no_majority_11u"
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
            self._rps = max(self._min_rps, self._rps * 0.7)

            if retry_after_seconds is not None:
                extra = max(0.0, float(retry_after_seconds))
            else:
                extra = min(8.0, 1.0 * self._consecutive_429)

            self._cooldown_until = max(self._cooldown_until, time.time() + extra)

    def note_success(self):
        with self._lock:
            if self._consecutive_429 > 0:
                self._consecutive_429 = max(0, self._consecutive_429 - 1)
            self._rps = min(self._max_rps, self._rps + 0.2)


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
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "few_shot")
PRED_DIR = os.path.join(OUTPUT_ROOT, "predictions")
FIG_DIR = os.path.join(OUTPUT_ROOT, "figures")
METRIC_DIR = os.path.join(OUTPUT_ROOT, "metrics")
DEBUG_DIR = os.path.join(OUTPUT_ROOT, "debug")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

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
    "sad": "sadness",
}
df["emotion"] = df["emotion"].map(lambda x: gold_map.get(x, x))

bad_gold = sorted(set(df["emotion"]) - set(ALLOWED_LABELS))
if bad_gold:
    raise RuntimeError(
        f"Gold labels contain values not in ALLOWED_LABELS: {bad_gold}\nAllowed: {ALLOWED_LABELS}"
    )

df["emotion"].value_counts().to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__DATASET_class_distribution.csv"),
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
# FEW-SHOT EXAMPLES
# ======================================================
FEW_SHOTS = [
    {"text": normalize_arabic("اليوم فرحت كثير لما نجحت بالمقابله"), "label": "joy"},
    {"text": normalize_arabic("مستفز جدا وما عنده اي احترام"), "label": "anger"},
    {"text": normalize_arabic("حاسه بضيق وزعل من اللي صار"), "label": "sadness"},
    {"text": normalize_arabic("خايفه يصير اشي سيء بكرا"), "label": "fear"},
    {"text": normalize_arabic("رحت السوق واشتريت خبز وحليب"), "label": "neutral"},
]


# ======================================================
# PROMPT / INSTRUCTIONS
# ======================================================
INSTRUCTIONS = (
    "You are an Arabic emotion classification engine. "
    "Classify the Arabic sentence into exactly one of these labels: "
    "joy, anger, sadness, fear, neutral. "
    "Return a strict JSON object with exactly one key: label. "
    "Do not return explanations or extra text."
)

def build_few_shot_prompt(text: str) -> str:
    parts = [
        "Choose exactly ONE label from:",
        "joy, anger, sadness, fear, neutral",
        "",
        "Output must be valid JSON only.",
        'Use exactly this schema: {"label":"joy"}',
        "Lowercase only.",
        "Do not output unknown.",
        "",
        "Examples:",
    ]

    for ex in FEW_SHOTS:
        parts.append(f'Sentence: "{ex["text"]}"')
        parts.append(f'Output: {{"label":"{ex["label"]}"}}')

    parts.append("")
    parts.append("Now classify this sentence:")
    parts.append(f'Sentence: "{text}"')

    return "\n".join(parts)


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
        "additionalProperties": False,
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
    if model_id in REASONING_MODELS:
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

def _max_retries_for_model(model_id: str) -> int:
    return MAX_RETRIES_BY_MODEL.get(model_id, 3)


# ======================================================
# CALL OPENAI RESPONSES API
# ======================================================
def call_model(model_id, model_name, text, temperature, limiter: AdaptiveRateLimiter, sample_i=None):
    prompt = build_few_shot_prompt(text)
    salt = f"{PROVIDER_NAME}|{model_name}|{sample_i}"
    last_err = None
    attempts_used = 0

    # ثابت وصغير، لا نكبره مع retries
    max_out = MAX_OUTPUT_TOKENS_BY_MODEL.get(model_id, 20)

    for attempt in range(1, _max_retries_for_model(model_id) + 1):
        attempts_used = attempt

        try:
            limiter.wait()
            if REQUEST_SPACING > 0:
                time.sleep(REQUEST_SPACING)

            kwargs = {
                "model": model_id,
                "input": prompt,
                "instructions": INSTRUCTIONS,
                "text": _text_config_for_model(model_id),
                "max_output_tokens": max_out,
            }

            reasoning_cfg = _reasoning_config_for_model(model_id)
            if reasoning_cfg is not None:
                kwargs["reasoning"] = reasoning_cfg

            if model_id not in NO_TEMPERATURE_MODELS:
                kwargs["temperature"] = float(temperature)

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

                if attempt < _max_retries_for_model(model_id):
                    _sleep_backoff(attempt, salt)
                    continue
                return "", 1, last_err, attempts_used, status, str(inc), str(err_obj)

            limiter.note_success()

            raw = extract_response_text(resp)
            if not raw.strip():
                err_obj = getattr(resp, "error", None)
                inc = getattr(resp, "incomplete_details", None)
                last_err = (
                    f"{UNK_EMPTY_MESSAGE}__status={status}"
                    f"__incomplete={str(inc)[:160]}__error={str(err_obj)[:160]}"
                )
                if attempt < _max_retries_for_model(model_id):
                    _sleep_backoff(attempt, salt)
                    continue
                return "", 1, last_err, attempts_used, status, str(inc), str(err_obj)

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

            if _is_transient_status(status) and attempt < _max_retries_for_model(model_id):
                if status == 429 and retry_after is not None and retry_after > 0:
                    time.sleep(min(15.0, retry_after))
                _sleep_backoff(attempt, salt)
                continue

            return "", 1, last_err, attempts_used, None, None, None

    return "", 1, (last_err or "max_retries_exceeded"), attempts_used, None, None, None


# ======================================================
# VOTING LOGIC
# ======================================================
def aggregate_votes(votes):
    if VOTES == 1:
        pred = votes[0]
        if pred == UNKNOWN_LABEL:
            return UNKNOWN_LABEL, "0", 1, UNK_ALL_FAILED
        return pred, "1", 0, UNK_OK

    unknown_votes = sum(1 for v in votes if v == UNKNOWN_LABEL)
    valid_votes = [v for v in votes if v != UNKNOWN_LABEL]

    if len(valid_votes) == 0:
        return UNKNOWN_LABEL, "000", unknown_votes, UNK_ALL_FAILED

    cnt = Counter(valid_votes)
    most_common = cnt.most_common()

    if most_common[0][1] >= 2:
        winner = most_common[0][0]
        pattern = "".join("1" if v == winner else "0" for v in votes)
        return winner, pattern, unknown_votes, UNK_OK

    if len(valid_votes) == 3 and len(set(valid_votes)) == 3:
        return UNKNOWN_LABEL, "111", unknown_votes, UNK_ALL_DIFFERENT_111

    if len(valid_votes) == 2 and len(set(valid_votes)) == 2 and unknown_votes == 1:
        return UNKNOWN_LABEL, "11u", unknown_votes, UNK_NO_MAJORITY_11U

    return UNKNOWN_LABEL, "???", unknown_votes, UNK_NO_MAJORITY_11U


def classify_sample(model_id, model_name, text, limiter: AdaptiveRateLimiter, sample_i=None):
    raw_list = []
    pred_list = []
    total_err_calls = 0
    total_calls_made = 0

    for v in range(VOTES):
        raw, err_flag, err_text, calls_made, resp_status, inc_details, err_obj = call_model(
            model_id=model_id,
            model_name=model_name,
            text=text,
            temperature=TEMP_VOTES,
            limiter=limiter,
            sample_i=f"{sample_i}_vote{v+1}",
        )

        total_calls_made += int(calls_made)

        if err_flag == 1:
            total_err_calls += 1
            pred = UNKNOWN_LABEL
        else:
            pred, _ = parse_json_label_strict_with_reason(raw)

        raw_list.append(raw if raw else "")
        pred_list.append(pred)

    final_pred, vote_pattern, unknown_votes, maj_reason = aggregate_votes(pred_list)
    abstained = 1 if final_pred == UNKNOWN_LABEL else 0

    if final_pred != UNKNOWN_LABEL:
        final_reason = UNK_OK
    else:
        final_reason = UNK_ALL_FAILED if all(p == UNKNOWN_LABEL for p in pred_list) else maj_reason

    raw_joined = " ||| ".join(raw_list)
    votes_list_json = json.dumps(pred_list, ensure_ascii=False)

    return (
        final_pred,
        total_err_calls,
        abstained,
        total_calls_made,
        raw_joined,
        vote_pattern,
        unknown_votes,
        final_reason,
        votes_list_json,
    )


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
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ======================================================
# PREFLIGHT MODEL TEST
# ======================================================
def test_models():
    print("\n[TEST] Testing each model with 1 request...")
    limiter = AdaptiveRateLimiter(initial_rps=1.0, min_rps=0.5, max_rps=3.0)
    test_text = normalize_arabic("اليوم زعلانه لان فيه غيوم وما نزل مطر")

    debug_rows = []

    for name, mid in MODELS.items():
        raw, err_flag, err_text, calls, status, incomplete_details, err_obj = call_model(
            mid, name, test_text, temperature=TEMP_VOTES, limiter=limiter, sample_i="TEST"
        )

        parsed, parse_reason = parse_json_label_strict_with_reason(raw) if raw else (UNKNOWN_LABEL, "no_raw")

        print("-----")
        print(f"Model: {name} / {mid}")
        print(f"err_flag: {err_flag} | calls: {calls}")
        print(f"status: {status}")
        print(f"err_text: {err_text}")
        print(f"incomplete_details: {incomplete_details}")
        print(f"error_obj: {err_obj}")
        print(f"raw_head: {(raw[:200] if raw else '<<EMPTY>>')}")
        print(f"parsed: {parsed} | parse_reason: {parse_reason}")
        print("-----")

        debug_rows.append({
            "model_name": name,
            "model_id": mid,
            "err_flag": err_flag,
            "calls": calls,
            "status": status,
            "err_text": err_text,
            "incomplete_details": incomplete_details,
            "error_obj": err_obj,
            "raw": raw,
            "parsed": parsed,
            "parse_reason": parse_reason,
        })

    pd.DataFrame(debug_rows).to_csv(
        os.path.join(DEBUG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__PREFLIGHT_DEBUG.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# ======================================================
# RUN BENCHMARK
# ======================================================
def run_benchmark():
    summary_rows = []

    for MODEL_NAME, MODEL_ID in MODELS.items():
        print(f"\nRunning {MODEL_NAME} (VOTES={VOTES}, TEMP={TEMP_VOTES})...")

        LIMITER = AdaptiveRateLimiter(
            initial_rps=INITIAL_RPS,
            min_rps=MIN_RPS,
            max_rps=MAX_RPS,
        )

        n = len(df)
        preds = [None] * n

        abstained_arr = [0] * n
        vote_patterns = [None] * n
        unknown_votes_arr = [0] * n
        unknown_reasons = [None] * n
        calls_made_arr = [0] * n
        votes_list_arr = [None] * n

        err_calls_total = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    classify_sample,
                    MODEL_ID,
                    MODEL_NAME,
                    df["text"].iloc[i],
                    LIMITER,
                    i,
                ): i
                for i in range(n)
            }

            for fut in tqdm(as_completed(futures), total=n, desc=MODEL_NAME):
                i = futures[fut]

                try:
                    pred, err_calls, abst_inc, calls_made, raw, pat, unk_votes, reason, votes_list_json = fut.result()
                except Exception as e:
                    pred, err_calls, abst_inc, calls_made, raw, pat, unk_votes, reason, votes_list_json = (
                        UNKNOWN_LABEL, VOTES, 1, 0, "", "000", VOTES,
                        f"{UNK_FUTURE_EXCEPTION}__{str(e)[:200]}",
                        json.dumps([UNKNOWN_LABEL] * VOTES, ensure_ascii=False),
                    )

                preds[i] = pred
                abstained_arr[i] = abst_inc
                vote_patterns[i] = pat
                unknown_votes_arr[i] = int(unk_votes)
                unknown_reasons[i] = reason
                calls_made_arr[i] = int(calls_made)
                votes_list_arr[i] = votes_list_json

                err_calls_total += err_calls

                if i < PRINT_FIRST_N:
                    with _print_lock:
                        print("-----")
                        print(
                            f"[{i+1}/{n}] GOLD={df['emotion'].iloc[i]}  "
                            f"PRED={pred}  abst={abst_inc}  reason={reason}  calls={calls_made}"
                        )
                        print(f"TEXT: {df['text'].iloc[i]}")
                        print(f"RAW : {raw if raw else '<<EMPTY>>'}")
                        print(f"VOTES: {votes_list_json}")
                        print("-----\n")

        y_true = df["emotion"].tolist()
        y_pred = preds

        unknown_total = y_pred.count(UNKNOWN_LABEL)
        unknown_rate = unknown_total / len(y_pred)
        coverage = 1 - unknown_rate

        pd.Series(unknown_reasons).value_counts(dropna=False).to_csv(
            os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__UNKNOWN_REASON_DISTRIBUTION.csv"),
            encoding="utf-8-sig",
        )

        y_pred_strict = [p if p != UNKNOWN_LABEL else STRICT_UNKNOWN_SENTINEL for p in y_pred]
        strict_accuracy = accuracy_score(y_true, y_pred_strict)
        strict_macro_f1 = f1_score(y_true, y_pred_strict, average="macro", labels=ALLOWED_LABELS, zero_division=0)
        strict_micro_f1 = f1_score(y_true, y_pred_strict, average="micro", labels=ALLOWED_LABELS, zero_division=0)
        strict_weighted_f1 = f1_score(y_true, y_pred_strict, average="weighted", labels=ALLOWED_LABELS, zero_division=0)

        report_strict_5 = classification_report(
            y_true, y_pred_strict, labels=ALLOWED_LABELS, output_dict=True, zero_division=0
        )
        pd.DataFrame(report_strict_5).transpose().to_csv(
            os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__REPORT_STRICT_5LABELS.csv"),
            encoding="utf-8-sig",
        )

        cm_with_unknown = confusion_matrix(y_true, y_pred, labels=STRICT_LABELS)
        save_confusion_matrix(
            cm_with_unknown,
            STRICT_LABELS,
            title=f"{MODEL_NAME} Confusion Matrix (incl. unknown)",
            outpath=os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_WITH_UNKNOWN.png"),
        )

        cm_5 = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
        save_confusion_matrix(
            cm_5,
            ALLOWED_LABELS,
            title=f"{MODEL_NAME} Confusion Matrix (5 labels)",
            outpath=os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_5LABELS.png"),
        )

        mask = [p != UNKNOWN_LABEL for p in y_pred]
        y_true_cov = [yt for yt, m in zip(y_true, mask) if m]
        y_pred_cov = [yp for yp in y_pred if yp != UNKNOWN_LABEL]

        if len(y_true_cov) > 0:
            covered_accuracy = accuracy_score(y_true_cov, y_pred_cov)
            covered_macro_f1 = f1_score(y_true_cov, y_pred_cov, average="macro", labels=COVERED_LABELS, zero_division=0)
            covered_micro_f1 = f1_score(y_true_cov, y_pred_cov, average="micro", labels=COVERED_LABELS, zero_division=0)
            covered_weighted_f1 = f1_score(
                y_true_cov, y_pred_cov, average="weighted", labels=COVERED_LABELS, zero_division=0
            )
        else:
            covered_accuracy = covered_macro_f1 = covered_micro_f1 = covered_weighted_f1 = 0.0

        out_pred_df = pd.DataFrame({
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
            "model_id": MODEL_ID,
            "run_id": RUN_ID,
            "votes": VOTES,
            "temp_votes": TEMP_VOTES,
            "max_workers": MAX_WORKERS,
            "votes_list": votes_list_arr,
        })

        out_pred_df = out_pred_df[UNIFIED_PRED_COLS]
        out_pred_df.to_csv(
            os.path.join(PRED_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__PREDICTIONS.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        reason_counter = Counter(unknown_reasons)
        summary_rows.append({
            "provider": PROVIDER_NAME,
            "model_name": MODEL_NAME,
            "model_id": MODEL_ID,
            "run_id": RUN_ID,
            "setup": "few-shot + single inference (V1) + OpenAI Responses API + structured JSON",
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

            "unknown_reason_all_failed": int(reason_counter.get(UNK_ALL_FAILED, 0)),
            "unknown_reason_all_different_111": int(reason_counter.get(UNK_ALL_DIFFERENT_111, 0)),
            "unknown_reason_no_majority_11u": int(reason_counter.get(UNK_NO_MAJORITY_11U, 0)),
            "unknown_reason_invalid_json_no_object": int(reason_counter.get(UNK_INVALID_JSON_NO_OBJECT, 0)),
            "unknown_reason_invalid_json_parse_error": int(reason_counter.get(UNK_INVALID_JSON_PARSE_ERROR, 0)),
            "unknown_reason_invalid_json_missing_label": int(reason_counter.get(UNK_INVALID_JSON_MISSING_LABEL, 0)),
            "unknown_reason_invalid_label": int(reason_counter.get(UNK_INVALID_LABEL, 0)),
            "unknown_reason_empty_message": int(reason_counter.get(UNK_EMPTY_MESSAGE, 0)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\nOpenAI few-shot benchmark completed successfully for all manuscript models.")
    print("RUN_ID:", RUN_ID)
    print("Models run:", list(MODELS.keys()))
    print("Outputs saved under:")
    print(" - predictions:", PRED_DIR)
    print(" - figures    :", FIG_DIR)
    print(" - metrics    :", METRIC_DIR)
    print(" - debug      :", DEBUG_DIR)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    test_models()
    run_benchmark()