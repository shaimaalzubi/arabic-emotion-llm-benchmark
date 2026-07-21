import os
import re
import time
import json
import random
import hashlib
import threading
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import matplotlib
matplotlib.use("Agg")

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from typing import Optional


# ===============================
# CONFIG
# ===============================
VOTES = 1
TEMP_VOTES = 0.0

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"

STRICT_UNKNOWN_SENTINEL = "unknown"
PROVIDER_NAME = "Google"

# Google-hosted models reported in the manuscript.
MODEL_CONFIGS = {
    "Gemini-2.5-Flash": {
        "model_id": "models/gemini-2.5-flash",
        "use_structured_output": True,
        "thinking_budget": 128,
    },
    "Gemma-3-12B": {
        # Historical API identifier used for the reported benchmark.
        # Availability may depend on account, region, and current API catalogue.
        "model_id": "models/gemma-3-12b-it",
        "use_structured_output": False,
        "thinking_budget": None,
    },
}

# ===============================
# SPEED / RELIABILITY (ANTI-HANG)
# ===============================
MAX_WORKERS = 2

MAX_RETRIES = 4
BASE_BACKOFF = 1.2
JITTER = 0.40

REQUEST_SPACING = 0.0

# timeout split:
# connect timeout = 20 sec
# read timeout    = 60 sec
CONNECT_TIMEOUT = 20
READ_TIMEOUT = 60

INITIAL_RPS = 0.35
MIN_RPS = 0.10
MAX_RPS = 0.60

MAX_OUTPUT_TOKENS = 512

SEED = 42
random.seed(SEED)
DETERMINISTIC_JITTER = True

STORE_VOTES_LIST = True
PRINT_FIRST_N = 10

RETRY_ON_INVALID_LABEL = True
RETRY_ON_PARSE_FAILURE = True
RETRY_ON_EMPTY_CANDIDATE = True

NO_PROGRESS_HEARTBEAT_SEC = 30

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__FEWSHOT_ANTIHANG"

UNIFIED_PRED_COLS = [
    "text", "gold_label", "prediction",
    "abstained", "vote_pattern", "unknown_votes", "unknown_reason", "calls_made",
    "provider", "model", "model_id", "run_id",
    "votes", "temp_votes", "max_workers",
]

# unknown reasons
UNK_API_ERROR = "api_error"
UNK_HTTP_STATUS = "http_status"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"
UNK_OK = "ok"
UNK_REGEX_LABEL = "regex_label"
UNK_REPAIRED_TRUNC = "repaired_truncation"
UNK_EMPTY_CANDIDATE = "empty_candidate_text"
UNK_RETRY_FIXED_INVALID_LABEL = "retry_fixed_invalid_label"
UNK_NON_JSON_PREFACE = "non_json_preface"
UNK_RETRY_FIXED_PARSE_FAILURE = "retry_fixed_parse_failure"
UNK_PROMPT_BLOCKED = "prompt_blocked"
UNK_MAX_TOKENS = "max_tokens"
UNK_REQUEST_TIMEOUT = "request_timeout"
UNK_EXCEPTION = "exception"


# ======================================================
# RATE LIMITER
# ======================================================
class AdaptiveRateLimiter:
    def __init__(self, initial_rps: float, min_rps: float, max_rps: float):
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

            extra = float(retry_after_seconds) if retry_after_seconds is not None else min(
                30.0, 3.0 * self._consecutive_429
            )
            self._cooldown_until = max(self._cooldown_until, time.time() + extra)

    def note_success(self):
        with self._lock:
            if self._consecutive_429 > 0:
                self._consecutive_429 = max(0, self._consecutive_429 - 1)
            self._rps = min(self._max_rps, self._rps + 0.02)

    def snapshot(self):
        with self._lock:
            return {
                "rps": round(self._rps, 3),
                "cooldown_until": self._cooldown_until,
                "consecutive_429": self._consecutive_429,
                "last_ts": self._last_ts,
            }


# ======================================================
# Helpers
# ======================================================
def _det_jitter(seed: int, attempt: int, salt: str) -> float:
    key = f"{seed}|{attempt}|{salt}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    val = int(h[:8], 16) / float(16 ** 8)
    return val * JITTER

def _sleep_backoff(attempt: int, salt: str):
    base = BASE_BACKOFF * (2 ** (attempt - 1))
    jitter = _det_jitter(SEED, attempt, salt) if DETERMINISTIC_JITTER else random.uniform(0, JITTER)
    time.sleep(base + jitter)

def normalize_arabic(text):
    text = str(text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _strip_code_fences(text: str) -> str:
    t = "" if text is None else str(text)
    t = re.sub(r"```[^\n]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```", "", t)
    return t.strip()

def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    t = _strip_code_fences(text)

    if t.startswith("{") and t.endswith("}"):
        return t

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
        repaired = blob.replace("'", '"')
        return json.loads(repaired)
    except Exception:
        return None

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

def _regex_label_anywhere(raw: str) -> Optional[str]:
    if not raw:
        return None
    low = _strip_code_fences(raw).lower()

    m = re.search(r"""\blabel\b\s*[:=]\s*["']?\s*(joy|anger|sadness|fear|neutral)\s*["']?""", low)
    if m:
        return m.group(1)

    m2 = re.search(r"""\b(emotion|prediction|class)\b\s*[:=]\s*["']?\s*(joy|anger|sadness|fear|neutral)\s*["']?""", low)
    if m2:
        return m2.group(2)

    hits = []
    for lab in ALLOWED_LABELS:
        if re.search(rf"\b{lab}\b", low):
            hits.append(lab)
    if len(hits) == 1:
        return hits[0]
    return None

def repair_truncated_label_json(raw: str) -> Optional[str]:
    if not raw:
        return None
    t = raw.strip()

    if "}" in t and "{" in t:
        return None

    m = re.search(r'\{\s*"\s*label\s*"\s*:\s*"\s*([a-zA-Z]{1,20})\s*$', t)
    if not m:
        return None

    prefix = m.group(1).lower()
    matches = [lab for lab in ALLOWED_LABELS if lab.startswith(prefix)]
    if len(matches) == 1:
        return json.dumps({"label": matches[0]})
    return None

def parse_label_with_reason(raw_text: str):
    raw = str(raw_text).strip() if raw_text is not None else ""
    if raw == "":
        return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT

    blob = _extract_first_json_object(raw)
    if blob is not None:
        obj = _try_load_json_obj(blob)
        if obj is None or not isinstance(obj, dict):
            lab = _regex_label_anywhere(raw)
            if lab:
                return lab, UNK_REGEX_LABEL
            return UNKNOWN_LABEL, UNK_INVALID_JSON_PARSE_ERROR

        label_key = None
        for k in obj.keys():
            if str(k).strip().lower() == "label":
                label_key = k
                break

        if label_key is None:
            lab = _regex_label_anywhere(raw)
            if lab:
                return lab, UNK_REGEX_LABEL
            return UNKNOWN_LABEL, UNK_INVALID_JSON_MISSING_LABEL

        lab = clean_label(obj.get(label_key))
        if lab == UNKNOWN_LABEL:
            lab2 = _regex_label_anywhere(raw)
            if lab2:
                return lab2, UNK_REGEX_LABEL
            return UNKNOWN_LABEL, UNK_INVALID_LABEL

        return lab, UNK_OK

    fixed = repair_truncated_label_json(raw)
    if fixed:
        obj2 = _try_load_json_obj(fixed)
        if isinstance(obj2, dict) and "label" in obj2:
            lab2 = clean_label(obj2.get("label"))
            if lab2 in ALLOWED_LABELS:
                return lab2, UNK_REPAIRED_TRUNC

    lab = _regex_label_anywhere(raw)
    if lab:
        return lab, UNK_REGEX_LABEL

    return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT


# ======================================================
# API + Session
# ======================================================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("ERROR: GOOGLE_API_KEY not found in .env")

SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=50,
    pool_maxsize=50,
    max_retries=0
)
SESSION.mount("https://", adapter)
SESSION.headers.update({"Content-Type": "application/json"})


# ======================================================
# Paths + Data
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


# ======================================================
# Prompts
# ======================================================
SYSTEM_MSG = (
    'Return only JSON: {"label":"joy"} '
    'Allowed labels: joy, anger, sadness, fear, neutral.'
)

SYSTEM_MSG_RETRY = (
    'ONLY return one JSON object exactly like {"label":"neutral"}. '
    'Allowed labels only: joy, anger, sadness, fear, neutral.'
)

SYSTEM_MSG_INVALID_LABEL_RETRY = (
    'Previous label invalid. Return only JSON with one valid label from: '
    'joy, anger, sadness, fear, neutral.'
)

SYSTEM_MSG_PARSE_FAIL_RETRY = (
    'Previous output invalid or truncated. Return only one complete JSON object.'
)

FEW_SHOT_EXAMPLES = """
Sentence: "فرحت كثير لما نجحت"
Output: {"label":"joy"}

Sentence: "انا معصب من اللي صار"
Output: {"label":"anger"}

Sentence: "حاسه بحزن شديد اليوم"
Output: {"label":"sadness"}

Sentence: "خفت لما سمعت الصوت"
Output: {"label":"fear"}

Sentence: "كان يوم عادي وما صار شيء"
Output: {"label":"neutral"}
""".strip()

PROMPT_BASE = """
Examples:
{FEW_SHOT_EXAMPLES}

Sentence: "{TEXT}"
""".strip()


# ======================================================
# Gemini helpers
# ======================================================
def extract_gemini_candidate_text(data: dict) -> str:
    cands = data.get("candidates", []) or []
    chunks = []

    for c in cands:
        content = (c or {}).get("content", {}) or {}
        parts = content.get("parts", []) or []
        for p in parts:
            if isinstance(p, dict):
                t = p.get("text")
                if t is not None:
                    t = str(t)
                    if t.strip():
                        chunks.append(t)

    return "".join(chunks).strip()

def inspect_gemini_response(data: dict):
    prompt_feedback = data.get("promptFeedback", {}) or {}
    block_reason = prompt_feedback.get("blockReason")

    cands = data.get("candidates", []) or []
    finish_reasons = []
    finish_messages = []

    for c in cands:
        if isinstance(c, dict):
            fr = c.get("finishReason")
            fm = c.get("finishMessage")
            if fr is not None:
                finish_reasons.append(str(fr))
            if fm is not None:
                finish_messages.append(str(fm))

    usage = data.get("usageMetadata", {}) or {}

    return {
        "block_reason": block_reason,
        "finish_reasons": finish_reasons,
        "finish_messages": finish_messages,
        "usage": usage,
    }

def get_thinking_config(model_cfg: dict):
    budget = model_cfg.get("thinking_budget")
    if budget is None:
        return None
    return {"thinkingBudget": int(budget)}


# ======================================================
# RUN
# ======================================================
summary_rows = []

for MODEL_NAME, MODEL_CFG in MODEL_CONFIGS.items():
    MODEL_ID = MODEL_CFG["model_id"]
    print(f"\nRunning {MODEL_NAME} (FEW-SHOT, VOTES=1)...")
    print(
        f"Runtime config: workers={MAX_WORKERS}, "
        f"initial_rps={INITIAL_RPS}, min_rps={MIN_RPS}, max_rps={MAX_RPS}, "
        f"connect_timeout={CONNECT_TIMEOUT}, read_timeout={READ_TIMEOUT}, "
        f"max_output_tokens={MAX_OUTPUT_TOKENS}"
    )

    URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_ID}:generateContent?key={API_KEY}"
    LIMITER = AdaptiveRateLimiter(INITIAL_RPS, MIN_RPS, MAX_RPS)
    THINKING_CONFIG = get_thinking_config(MODEL_CFG)
    USE_STRUCTURED_OUTPUT = bool(MODEL_CFG.get("use_structured_output", False))

    active_lock = threading.Lock()
    active_requests = set()

    def _mark_active(sample_i):
        with active_lock:
            active_requests.add(str(sample_i))

    def _mark_done(sample_i):
        with active_lock:
            active_requests.discard(str(sample_i))

    def _active_count():
        with active_lock:
            return len(active_requests)

    def _active_preview(max_items=8):
        with active_lock:
            arr = list(active_requests)
        arr = sorted(arr)[:max_items]
        return arr

    def _post_once(text: str, system_msg: str, temperature: float, sample_i=None):
        prompt = f"{system_msg}\n{text}"

        generation_config = {
            "temperature": float(temperature),
            "maxOutputTokens": int(MAX_OUTPUT_TOKENS),
            "candidateCount": 1,
        }

        if USE_STRUCTURED_OUTPUT:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = {
                "type": "OBJECT",
                "properties": {
                    "label": {
                        "type": "STRING",
                        "enum": ["joy", "anger", "sadness", "fear", "neutral"]
                    }
                },
                "required": ["label"]
            }

        if THINKING_CONFIG is not None:
            generation_config["thinkingConfig"] = THINKING_CONFIG

        body = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": generation_config,
        }

        LIMITER.wait()
        if REQUEST_SPACING > 0:
            time.sleep(REQUEST_SPACING)

        print(f"[REQ] sample={sample_i}", flush=True)
        _mark_active(sample_i)
        try:
            r = SESSION.post(
                URL,
                json=body,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            print(f"[RESP] sample={sample_i} status={r.status_code}", flush=True)
            return r
        finally:
            _mark_done(sample_i)

    def call_model(text: str, temperature: float, sample_i=None, force_system_msg: Optional[str] = None):
        salt = f"{PROVIDER_NAME}|{MODEL_NAME}|{sample_i}"
        last_err = None
        attempts_used = 0

        for attempt in range(1, MAX_RETRIES + 1):
            attempts_used = attempt
            system_msg = force_system_msg if force_system_msg is not None else (
                SYSTEM_MSG if attempt == 1 else SYSTEM_MSG_RETRY
            )

            try:
                short_text = (
                    PROMPT_BASE
                    .replace("{FEW_SHOT_EXAMPLES}", FEW_SHOT_EXAMPLES)
                    .replace("{TEXT}", text.replace('"', '\\"'))
                )

                r = _post_once(
                    short_text,
                    system_msg=system_msg,
                    temperature=temperature,
                    sample_i=sample_i
                )

                if r.status_code != 200:
                    if r.status_code in (408, 429, 500, 502, 503, 504):
                        if r.status_code == 429:
                            ra = r.headers.get("Retry-After")
                            ra_s = None
                            try:
                                ra_s = float(ra) if ra is not None else None
                            except Exception:
                                ra_s = None
                            LIMITER.note_429(retry_after_seconds=ra_s)

                        last_err = f"{UNK_HTTP_STATUS}_{r.status_code}__{r.text[:300]}"
                        print(f"[RETRY] sample={sample_i} attempt={attempt} err={last_err}", flush=True)
                        _sleep_backoff(attempt, salt)
                        continue

                    return "", 1, f"{UNK_HTTP_STATUS}_{r.status_code}__{r.text[:800]}", attempts_used

                data = r.json()
                info = inspect_gemini_response(data)

                if "error" in data:
                    msg = data.get("error", {}).get("message", "")
                    code = data.get("error", {}).get("code")
                    if code in (429, 500, 503):
                        if code == 429:
                            LIMITER.note_429()
                        last_err = f"{UNK_API_ERROR}_{code}__{str(msg)[:300]}"
                        print(f"[RETRY] sample={sample_i} attempt={attempt} err={last_err}", flush=True)
                        _sleep_backoff(attempt, salt)
                        continue
                    return "", 1, f"{UNK_API_ERROR}_{code}__{str(msg)[:800]}", attempts_used

                if info["block_reason"] is not None:
                    return "", 1, f"{UNK_PROMPT_BLOCKED}__{info['block_reason']}", attempts_used

                raw_text = extract_gemini_candidate_text(data)

                if not raw_text.strip():
                    fr = ",".join(info["finish_reasons"]) if info["finish_reasons"] else "NO_FINISH_REASON"
                    fm = " | ".join(info["finish_messages"]) if info["finish_messages"] else ""
                    last_err = f"{UNK_EMPTY_CANDIDATE}__finish={fr}__msg={fm}"

                    if "MAX_TOKENS" in fr:
                        last_err = f"{UNK_MAX_TOKENS}__{last_err}"

                    if RETRY_ON_EMPTY_CANDIDATE:
                        print(f"[RETRY] sample={sample_i} attempt={attempt} err={last_err}", flush=True)
                        _sleep_backoff(attempt, salt)
                        continue

                    return "", 1, last_err, attempts_used

                if "{" not in raw_text:
                    last_err = UNK_NON_JSON_PREFACE
                    print(f"[RETRY] sample={sample_i} attempt={attempt} err={last_err}", flush=True)
                    _sleep_backoff(attempt, salt)
                    continue

                LIMITER.note_success()
                return raw_text.strip(), 0, None, attempts_used

            except requests.exceptions.Timeout as e:
                last_err = f"{UNK_REQUEST_TIMEOUT}__{repr(e)[:400]}"
                print(f"[EXC] sample={sample_i} attempt={attempt} err={last_err}", flush=True)
                _sleep_backoff(attempt, salt)
                continue

            except requests.exceptions.ConnectionError as e:
                last_err = f"{UNK_EXCEPTION}__connection_error__{repr(e)[:400]}"
                print(f"[EXC] sample={sample_i} attempt={attempt} err={last_err}", flush=True)
                _sleep_backoff(attempt, salt)
                continue

            except Exception as e:
                last_err = f"{UNK_EXCEPTION}__{repr(e)[:500]}"
                print(f"[EXC] sample={sample_i} attempt={attempt} err={last_err}", flush=True)
                _sleep_backoff(attempt, salt)
                continue

        return "", 1, (last_err or "max_retries_exceeded"), attempts_used

    # ===============================
    # Test call
    # ===============================
    test_text = "انا مبسوطه اليوم"
    raw_t, err_t, err_msg_t, att_t = call_model(test_text, temperature=TEMP_VOTES, sample_i="TEST")
    print("[TEST] err_flag:", err_t, "| attempts:", att_t)
    if err_t == 1:
        print("[TEST] err:", err_msg_t)
        raise RuntimeError(f"Test call failed for {MODEL_NAME} in the few-shot anti-hang run.")
    else:
        print("[TEST] raw_head:", raw_t[:120])
        test_pred, test_reason = parse_label_with_reason(raw_t)
        print("[TEST] parsed:", test_pred, "| reason:", test_reason)
        if test_pred == UNKNOWN_LABEL:
            print("[TEST] raw_len:", len(raw_t))
            print("[TEST] raw_full:", repr(raw_t[:600]))
            raise RuntimeError("Test call returned non-parseable output. Check raw_head above.")

    # ===============================
    # PRINT FIRST N IN ORDER
    # ===============================
    printed_next = [0]
    pending_print = {}
    print_lock = threading.Lock()

    def _queue_print_line(i: int, gold: str, row: dict, text: str):
        if i >= PRINT_FIRST_N:
            return
        line = (
            f"-----\n"
            f"[{i:04d}] GOLD={gold:<7}  PRED={row['pred']:<7}  "
            f"abst={row['abstained']}  reason={row['unknown_reason']}  calls={row['calls_made']}\n"
            f"TEXT: {text}\n"
            f"RAW : {repr(row['raw'])}\n"
        )
        with print_lock:
            pending_print[i] = line
            while printed_next[0] in pending_print:
                print(pending_print.pop(printed_next[0]), flush=True)
                printed_next[0] += 1

    def classify_one(i: int):
        text = df.loc[i, "text"]
        gold = df.loc[i, "emotion"]

        raw, err_flag, err_text, attempts_used = call_model(text, temperature=TEMP_VOTES, sample_i=i)
        calls_made = int(attempts_used)

        if err_flag == 1:
            reason = UNK_API_ERROR
            if isinstance(err_text, str):
                if err_text.startswith(f"{UNK_PROMPT_BLOCKED}__"):
                    reason = UNK_PROMPT_BLOCKED
                elif err_text.startswith(f"{UNK_MAX_TOKENS}__"):
                    reason = UNK_MAX_TOKENS
                elif err_text.startswith(f"{UNK_EMPTY_CANDIDATE}__"):
                    reason = UNK_EMPTY_CANDIDATE
                elif err_text.startswith(f"{UNK_REQUEST_TIMEOUT}__"):
                    reason = UNK_REQUEST_TIMEOUT
                elif err_text.startswith(f"{UNK_HTTP_STATUS}_"):
                    reason = UNK_HTTP_STATUS
                elif err_text.startswith(f"{UNK_EXCEPTION}__"):
                    reason = UNK_EXCEPTION

            row = {
                "pred": UNKNOWN_LABEL,
                "err_calls": int(attempts_used),
                "abstained": 1,
                "calls_made": calls_made,
                "vote_pattern": "0",
                "unknown_votes": 1,
                "unknown_reason": reason,
                "raw": f"__API_ERROR__ {str(err_text)[:600]}",
            }
            return i, gold, text, row

        pred, parse_reason = parse_label_with_reason(raw)

        if RETRY_ON_PARSE_FAILURE and pred == UNKNOWN_LABEL and parse_reason in (
            UNK_INVALID_JSON_NO_OBJECT,
            UNK_INVALID_JSON_PARSE_ERROR,
            UNK_INVALID_JSON_MISSING_LABEL,
        ):
            raw2, err2, err_msg2, att2 = call_model(
                text,
                temperature=TEMP_VOTES,
                sample_i=f"{i}__PARSE_RETRY",
                force_system_msg=SYSTEM_MSG_PARSE_FAIL_RETRY,
            )
            calls_made += int(att2)

            if err2 == 0:
                pred2, reason2 = parse_label_with_reason(raw2)
                if pred2 != UNKNOWN_LABEL:
                    row = {
                        "pred": pred2,
                        "err_calls": 0,
                        "abstained": 0,
                        "calls_made": calls_made,
                        "vote_pattern": "1",
                        "unknown_votes": 0,
                        "unknown_reason": UNK_RETRY_FIXED_PARSE_FAILURE,
                        "raw": raw2[:600],
                    }
                    return i, gold, text, row
                else:
                    raw = raw2
                    parse_reason = reason2
                    pred = pred2
            else:
                row = {
                    "pred": UNKNOWN_LABEL,
                    "err_calls": int(att2),
                    "abstained": 1,
                    "calls_made": calls_made,
                    "vote_pattern": "0",
                    "unknown_votes": 1,
                    "unknown_reason": UNK_API_ERROR,
                    "raw": f"__API_ERROR__ {str(err_msg2)[:600]}",
                }
                return i, gold, text, row

        if pred == UNKNOWN_LABEL and parse_reason == UNK_INVALID_LABEL and RETRY_ON_INVALID_LABEL:
            raw2, err2, err_msg2, att2 = call_model(
                text,
                temperature=TEMP_VOTES,
                sample_i=f"{i}__INVLBL_RETRY",
                force_system_msg=SYSTEM_MSG_INVALID_LABEL_RETRY,
            )
            calls_made += int(att2)

            if err2 == 0:
                pred2, reason2 = parse_label_with_reason(raw2)
                if pred2 != UNKNOWN_LABEL:
                    row = {
                        "pred": pred2,
                        "err_calls": 0,
                        "abstained": 0,
                        "calls_made": calls_made,
                        "vote_pattern": "1",
                        "unknown_votes": 0,
                        "unknown_reason": UNK_RETRY_FIXED_INVALID_LABEL,
                        "raw": raw2[:600],
                    }
                    return i, gold, text, row
                else:
                    raw = raw2
                    parse_reason = reason2
            else:
                row = {
                    "pred": UNKNOWN_LABEL,
                    "err_calls": int(att2),
                    "abstained": 1,
                    "calls_made": calls_made,
                    "vote_pattern": "0",
                    "unknown_votes": 1,
                    "unknown_reason": UNK_API_ERROR,
                    "raw": f"__API_ERROR__ {str(err_msg2)[:600]}",
                }
                return i, gold, text, row

        if pred == UNKNOWN_LABEL:
            row = {
                "pred": pred,
                "err_calls": 0,
                "abstained": 1,
                "calls_made": calls_made,
                "vote_pattern": "0",
                "unknown_votes": 1,
                "unknown_reason": parse_reason,
                "raw": raw[:600],
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
            "raw": raw[:600],
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

    completed_count = 0
    last_completed_index = None
    start_ts = time.time()
    last_progress_ts = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {executor.submit(classify_one, i): i for i in range(n)}
        remaining = set(future_to_index.keys())

        pbar = tqdm(total=n, desc=MODEL_NAME)

        while remaining:
            done, remaining = wait(
                remaining,
                timeout=NO_PROGRESS_HEARTBEAT_SEC,
                return_when=FIRST_COMPLETED
            )

            if not done:
                elapsed = time.time() - start_ts
                limiter_state = LIMITER.snapshot()
                print(
                    f"[HEARTBEAT] no completed sample for {NO_PROGRESS_HEARTBEAT_SEC}s | "
                    f"done={completed_count}/{n} | "
                    f"elapsed_sec={int(elapsed)} | "
                    f"active_requests={_active_count()} | "
                    f"active_preview={_active_preview()} | "
                    f"rps={limiter_state['rps']} | "
                    f"consecutive_429={limiter_state['consecutive_429']} | "
                    f"last_completed_index={last_completed_index}",
                    flush=True
                )
                continue

            for fut in done:
                i, gold, text, row = fut.result()

                preds[i] = row["pred"]
                abstained_arr[i] = int(row["abstained"])
                vote_patterns[i] = row["vote_pattern"]
                unknown_votes_arr[i] = int(row["unknown_votes"])
                unknown_reasons[i] = row["unknown_reason"]
                calls_made_arr[i] = int(row["calls_made"])
                raw_preview_arr[i] = repr(row["raw"])

                if STORE_VOTES_LIST:
                    votes_list_arr[i] = json.dumps([row["pred"]], ensure_ascii=False)

                error_calls_total += int(row["err_calls"])
                abstain_count += int(row["abstained"])

                if i < PRINT_FIRST_N:
                    _queue_print_line(i, gold, row, text)

                completed_count += 1
                last_completed_index = i
                last_progress_ts = time.time()
                pbar.update(1)

        pbar.close()

    # ===============================
    # Metrics
    # ===============================
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
        "model_id": MODEL_ID,
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
        "model_id": MODEL_ID,
        "structured_output": USE_STRUCTURED_OUTPUT,
        "thinking_budget": MODEL_CFG.get("thinking_budget"),
        "run_id": RUN_ID,
        "setup": "few-shot + single inference (ANTI-HANG) — Google manuscript models + model-specific output constraints + conservative speed + connect/read timeout + heartbeat + parse/invalid-label retries + empty-candidate handling",
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
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_FEWSHOT_ANTIHANG_strict_json.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nGoogle benchmark completed successfully.")
print("RUN_ID:", RUN_ID)
print("Models run:", list(MODEL_CONFIGS.keys()))
print("Outputs saved under:")
print(" - predictions:", PRED_DIR)
print(" - metrics    :", METRIC_DIR)