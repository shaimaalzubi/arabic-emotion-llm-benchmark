import os
import re
import time
import json
import random
import hashlib
import threading
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================
# CONFIG
# ======================================================
VOTES = 3
FIXED_TEMPERATURE = 0.3

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_LABELS = ALLOWED_LABELS + [UNKNOWN_LABEL]
COVERED_LABELS = ALLOWED_LABELS
STRICT_UNKNOWN_SENTINEL = "__unknown__"

PROVIDER_NAME = "Groq"

MAX_RETRIES = 4
BASE_BACKOFF = 0.8
JITTER = 0.25

SEED = 42
random.seed(SEED)
DETERMINISTIC_JITTER = True

STORE_VOTES_LIST = True
STORE_RAW_VOTES = True

DEBUG_PRINT_FIRST = 10
ENABLE_RESUME = True
SAVE_EVERY_N = 25
TEST_CALL_BEFORE_RUN = True

RETRY_ON_PARSE_FAILURE = True
RETRY_ON_INVALID_LABEL = True

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# ======================================================
# UNKNOWN REASONS
# ======================================================
UNK_OK = "ok"
UNK_API_ERROR = "api_error"
UNK_EMPTY_RESPONSE = "empty_response"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"
UNK_REGEX_LABEL = "regex_label"
UNK_REPAIRED_TRUNC = "repaired_truncation"
UNK_NON_JSON_PREFACE = "non_json_preface"

FINAL_MAJORITY = "majority"
FINAL_ALL_FAILED = "all_failed"
FINAL_ALL_UNPARSEABLE = "all_unparseable"
FINAL_ALL_DIFFERENT_111 = "all_different_111"
FINAL_NO_MAJORITY_11U = "no_majority_11u"


# ======================================================
# MODEL CONFIGS (PER KEY)
# ======================================================
MODEL_CONFIGS = {
    "Qwen-3-32B": {
        "model_id": "qwen/qwen3-32b",
        "max_workers_base": 4,
        "workers_per_key": 1,

        "per_key_initial_rps": 0.30,
        "per_key_min_rps": 0.10,
        "per_key_max_rps": 0.60,

        "max_output_tokens": 120,
        "request_spacing": 0.02,
        "connect_timeout": 20,
        "read_timeout": 60,
        "use_system_msg": True,
        "reasoning_effort": "none",
    },
    "Allam-2-7B": {
        "model_id": "allam-2-7b",
        "max_workers_base": 6,
        "workers_per_key": 1,

        "per_key_initial_rps": 0.16,
        "per_key_min_rps": 0.08,
        "per_key_max_rps": 0.22,

        "max_output_tokens": 20,
        "request_spacing": 0.02,
        "connect_timeout": 20,
        "read_timeout": 45,
        "use_system_msg": True,
        "reasoning_effort": None,
    },
    "LLaMA-3.1-8B": {
        "model_id": "llama-3.1-8b-instant",
        "max_workers_base": 8,
        "workers_per_key": 2,

        "per_key_initial_rps": 0.60,
        "per_key_min_rps": 0.20,
        "per_key_max_rps": 1.00,

        "max_output_tokens": 20,
        "request_spacing": 0.00,
        "connect_timeout": 20,
        "read_timeout": 60,
        "use_system_msg": True,
        "reasoning_effort": None,
    },
}


# ======================================================
# API KEYS
# ======================================================
load_dotenv()

def load_groq_keys():
    keys = []

    base_key = os.getenv("GROQ_API_KEY")
    if base_key:
        keys.append(base_key.strip())

    for i in range(1, 21):
        k = os.getenv(f"GROQ_API_KEY{i}")
        if k:
            keys.append(k.strip())

    seen = set()
    unique_keys = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return unique_keys


API_KEYS = load_groq_keys()

if not API_KEYS:
    raise RuntimeError(
        "No Groq API keys found in .env.\n"
        "Use e.g. GROQ_API_KEY1=..., GROQ_API_KEY2=..."
    )

CLIENTS = [OpenAI(api_key=k, base_url="https://api.groq.com/openai/v1") for k in API_KEYS]
NUM_KEYS = len(CLIENTS)
print(f"Loaded {NUM_KEYS} Groq API key(s).")


class ClientPool:
    def __init__(self, clients):
        self.clients = clients
        self.n = len(clients)
        self.idx = 0
        self.lock = threading.Lock()

    def next_client(self):
        with self.lock:
            c = self.clients[self.idx]
            self.idx = (self.idx + 1) % self.n
            return c


CLIENT_POOL = ClientPool(CLIENTS)


# ======================================================
# BUILD RUNTIME CONFIG
# ======================================================
def build_runtime_config(cfg, num_keys: int):
    max_workers = cfg["max_workers_base"] + cfg["workers_per_key"] * max(0, num_keys - 1)

    initial_rps = cfg["per_key_initial_rps"] * num_keys
    min_rps = cfg["per_key_min_rps"] * num_keys
    max_rps = cfg["per_key_max_rps"] * num_keys

    max_workers = max(1, min(max_workers, 64))
    initial_rps = max(min_rps, min(initial_rps, max_rps))

    return {
        "model_id": cfg["model_id"],
        "max_workers": max_workers,
        "initial_rps": initial_rps,
        "min_rps": min_rps,
        "max_rps": max_rps,
        "temperature": FIXED_TEMPERATURE,
        "max_output_tokens": cfg["max_output_tokens"],
        "request_spacing": cfg["request_spacing"],
        "connect_timeout": cfg["connect_timeout"],
        "read_timeout": cfg["read_timeout"],
        "use_system_msg": cfg.get("use_system_msg", True),
        "reasoning_effort": cfg.get("reasoning_effort"),
    }


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
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "three_votes")
PRED_DIR = os.path.join(OUTPUT_ROOT, "predictions")
FIG_DIR = os.path.join(OUTPUT_ROOT, "figures")
METRIC_DIR = os.path.join(OUTPUT_ROOT, "metrics")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8")


# ======================================================
# DATA HYGIENE
# ======================================================
required_cols = {"text", "emotion"}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"CSV is missing required columns: {sorted(missing)}")

df = df.dropna(subset=["text", "emotion"]).reset_index(drop=True)
df["text"] = df["text"].astype(str)
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
        f"Gold labels contain values not in ALLOWED_LABELS: {bad_gold}\n"
        f"Allowed: {ALLOWED_LABELS}"
    )


# ======================================================
# NORMALIZATION
# ======================================================
def normalize_arabic(text: str) -> str:
    text = str(text)
    text = re.sub(r"[إأٱآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["text"] = df["text"].apply(normalize_arabic)

df["emotion"].value_counts().to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__DATASET_class_distribution.csv"),
    encoding="utf-8-sig"
)


# ======================================================
# PROMPTS
# ======================================================
SYSTEM_MSG = (
    "You are a STRICT emotion classifier.\n"
    "Return ONLY ONE valid JSON object.\n"
    "Use EXACTLY one key: label.\n"
    "The label MUST be one of: joy, anger, sadness, fear, neutral.\n"
    "Use ENGLISH lowercase labels ONLY.\n"
    "No Arabic labels.\n"
    "No explanation.\n"
    "No markdown.\n"
    "No text before or after the JSON.\n"
    "Reply MUST start with '{' and end with '}'."
)

SYSTEM_MSG_RETRY = (
    "INVALID OUTPUT.\n"
    "Return ONLY one JSON object exactly like {\"label\":\"neutral\"}.\n"
    "Allowed labels ONLY: joy, anger, sadness, fear, neutral.\n"
    "No other labels.\n"
    "No extra text.\n"
    "No markdown.\n"
    "Start with '{' and end with '}'."
)

SYSTEM_MSG_INVALID_LABEL_RETRY = (
    "YOUR LAST LABEL WAS INVALID.\n"
    "You MUST choose ONLY from: joy, anger, sadness, fear, neutral.\n"
    "Return ONLY JSON like {\"label\":\"joy\"}.\n"
    "Lowercase only.\n"
    "No extra text.\n"
    "Start with '{' and end with '}'."
)

SYSTEM_MSG_PARSE_FAIL_RETRY = (
    "YOUR LAST OUTPUT WAS TRUNCATED OR INVALID.\n"
    "Return ONLY one complete JSON object like {\"label\":\"neutral\"}.\n"
    "Allowed labels ONLY: joy, anger, sadness, fear, neutral.\n"
    "No extra text.\n"
    "No markdown.\n"
    "Start with '{' and end with '}'."
)

PROMPT_BASE = """
Classify the emotion of the following Arabic sentence.

Rules:
- Choose ONLY one label from: joy, anger, sadness, fear, neutral
- Output ONLY JSON: {"label":"joy"}
- Use ENGLISH lowercase labels ONLY
- DO NOT output Arabic words
- DO NOT explain
- If unsure, return {"label":"neutral"}

Sentence:
"{TEXT}"
""".strip()


# ======================================================
# ROBUST PARSER
# ======================================================
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


def clean_label(lbl) -> str:
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
        return UNKNOWN_LABEL, UNK_EMPTY_RESPONSE

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

    if "{" not in raw:
        return UNKNOWN_LABEL, UNK_NON_JSON_PREFACE

    return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT


# ======================================================
# BACKOFF
# ======================================================
def _is_transient_error(msg_lower: str) -> bool:
    return (
        ("rate" in msg_lower)
        or ("429" in msg_lower)
        or ("too many requests" in msg_lower)
        or ("overloaded" in msg_lower)
        or ("temporarily" in msg_lower and "unavailable" in msg_lower)
        or ("timeout" in msg_lower)
        or ("timed out" in msg_lower)
        or ("connection" in msg_lower)
        or ("server error" in msg_lower)
        or ("502" in msg_lower)
        or ("503" in msg_lower)
        or ("504" in msg_lower)
    )


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
# RATE LIMITER
# ======================================================
class AdaptiveRateLimiter:
    def __init__(self, initial_rps: float, min_rps: float, max_rps: float):
        self.current_rps = max(min(initial_rps, max_rps), min_rps)
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.lock = threading.Lock()
        self.next_allowed_time = 0.0

    def wait_for_slot(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed_time:
                sleep_for = self.next_allowed_time - now
                time.sleep(sleep_for)
                now = time.time()

            interval = 1.0 / max(self.current_rps, 1e-6)
            self.next_allowed_time = max(self.next_allowed_time, now) + interval

    def on_success(self):
        with self.lock:
            self.current_rps = min(self.max_rps, self.current_rps * 1.05 + 0.003)

    def on_transient_error(self):
        with self.lock:
            self.current_rps = max(self.min_rps, self.current_rps * 0.75)

    def get_rps(self) -> float:
        with self.lock:
            return self.current_rps


# ======================================================
# RESPONSE EXTRACTION
# ======================================================
def extract_chat_text(resp) -> str:
    try:
        content = resp.choices[0].message.content
        if content is None:
            return ""
        return str(content).strip()
    except Exception:
        return ""


# ======================================================
# CALL MODEL
# ======================================================
def call_model(
    model_id: str,
    model_name: str,
    text: str,
    temperature: float,
    max_output_tokens: int,
    request_spacing: float,
    connect_timeout: int,
    read_timeout: int,
    rate_limiter: AdaptiveRateLimiter,
    use_system_msg: bool = True,
    cfg_reasoning_effort: Optional[str] = None,
    sample_i=None,
    vote_i=None,
    force_system_msg: Optional[str] = None,
):
    prompt = PROMPT_BASE.replace("{TEXT}", text)
    salt = f"{PROVIDER_NAME}|{model_name}|{sample_i}|{vote_i}"
    last_err = None
    attempts_used = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt

        try:
            rate_limiter.wait_for_slot()

            if request_spacing > 0:
                time.sleep(request_spacing)

            client = CLIENT_POOL.next_client()

            chosen_system_msg = force_system_msg if force_system_msg is not None else (
                SYSTEM_MSG if attempt == 1 else SYSTEM_MSG_RETRY
            )

            if use_system_msg:
                messages = [
                    {"role": "system", "content": chosen_system_msg},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": chosen_system_msg + "\n\n" + prompt},
                ]

            request_kwargs = {
                "model": model_id,
                "temperature": float(temperature),
                "max_tokens": max_output_tokens,
                "messages": messages,
                "timeout": (connect_timeout, read_timeout),
            }

            reasoning_effort = cfg_reasoning_effort
            if reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = reasoning_effort

            resp = client.chat.completions.create(**request_kwargs)

            raw = extract_chat_text(resp)

            if not raw:
                rate_limiter.on_transient_error()
                last_err = UNK_EMPTY_RESPONSE
                _sleep_backoff(attempt, salt)
                continue

            if "{" not in raw:
                rate_limiter.on_transient_error()
                last_err = UNK_NON_JSON_PREFACE
                _sleep_backoff(attempt, salt)
                continue

            rate_limiter.on_success()
            return raw, 0, None, attempts_used

        except Exception as e:
            err = str(e).lower()
            last_err = str(e)[:300] if str(e) else e.__class__.__name__

            if _is_transient_error(err):
                rate_limiter.on_transient_error()
                _sleep_backoff(attempt, salt)
                continue

            return "", 1, f"{UNK_API_ERROR}__{type(e).__name__}__{last_err}", attempts_used

    return "", 1, (last_err or "max_retries_exceeded"), attempts_used


# ======================================================
# SINGLE VOTE RESOLUTION
# ======================================================
def resolve_single_vote(
    model_id: str,
    model_name: str,
    text: str,
    cfg: dict,
    rate_limiter: AdaptiveRateLimiter,
    sample_suffix: str,
):
    raw, err_flag, err_text, attempts_used = call_model(
        model_id=model_id,
        model_name=model_name,
        text=text,
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_output_tokens"],
        request_spacing=cfg["request_spacing"],
        connect_timeout=cfg["connect_timeout"],
        read_timeout=cfg["read_timeout"],
        rate_limiter=rate_limiter,
        use_system_msg=cfg["use_system_msg"],
        cfg_reasoning_effort=cfg.get("reasoning_effort"),
        sample_i=sample_suffix,
        vote_i=0,
    )

    calls_made = int(attempts_used)

    if err_flag == 1:
        return {
            "pred": UNKNOWN_LABEL,
            "reason": UNK_API_ERROR,
            "raw": f"__API_ERROR__ {str(err_text)[:600]}",
            "calls_made": calls_made,
            "err_calls": int(attempts_used),
            "call_status": "api_error",
        }

    pred, parse_reason = parse_label_with_reason(raw)

    if RETRY_ON_PARSE_FAILURE and pred == UNKNOWN_LABEL and parse_reason in (
        UNK_INVALID_JSON_NO_OBJECT,
        UNK_INVALID_JSON_PARSE_ERROR,
        UNK_INVALID_JSON_MISSING_LABEL,
        UNK_NON_JSON_PREFACE,
    ):
        raw2, err2, err_msg2, att2 = call_model(
            model_id=model_id,
            model_name=model_name,
            text=text,
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_output_tokens"],
            request_spacing=cfg["request_spacing"],
            connect_timeout=cfg["connect_timeout"],
            read_timeout=cfg["read_timeout"],
            rate_limiter=rate_limiter,
            use_system_msg=cfg["use_system_msg"],
            cfg_reasoning_effort=cfg.get("reasoning_effort"),
            sample_i=f"{sample_suffix}__PARSE_RETRY",
            vote_i=0,
            force_system_msg=SYSTEM_MSG_PARSE_FAIL_RETRY,
        )
        calls_made += int(att2)

        if err2 == 0:
            pred2, reason2 = parse_label_with_reason(raw2)
            if pred2 != UNKNOWN_LABEL:
                return {
                    "pred": pred2,
                    "reason": "retry_fixed_parse_failure",
                    "raw": raw2[:600],
                    "calls_made": calls_made,
                    "err_calls": 0,
                    "call_status": "ok_after_parse_retry",
                }
            raw = raw2
            pred = pred2
            parse_reason = reason2
        else:
            return {
                "pred": UNKNOWN_LABEL,
                "reason": UNK_API_ERROR,
                "raw": f"__API_ERROR__ {str(err_msg2)[:600]}",
                "calls_made": calls_made,
                "err_calls": int(att2),
                "call_status": "api_error_after_parse_retry",
            }

    if pred == UNKNOWN_LABEL and parse_reason == UNK_INVALID_LABEL and RETRY_ON_INVALID_LABEL:
        raw2, err2, err_msg2, att2 = call_model(
            model_id=model_id,
            model_name=model_name,
            text=text,
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_output_tokens"],
            request_spacing=cfg["request_spacing"],
            connect_timeout=cfg["connect_timeout"],
            read_timeout=cfg["read_timeout"],
            rate_limiter=rate_limiter,
            use_system_msg=cfg["use_system_msg"],
            cfg_reasoning_effort=cfg.get("reasoning_effort"),
            sample_i=f"{sample_suffix}__INVLBL_RETRY",
            vote_i=0,
            force_system_msg=SYSTEM_MSG_INVALID_LABEL_RETRY,
        )
        calls_made += int(att2)

        if err2 == 0:
            pred2, reason2 = parse_label_with_reason(raw2)
            if pred2 != UNKNOWN_LABEL:
                return {
                    "pred": pred2,
                    "reason": "retry_fixed_invalid_label",
                    "raw": raw2[:600],
                    "calls_made": calls_made,
                    "err_calls": 0,
                    "call_status": "ok_after_invalid_label_retry",
                }
            raw = raw2
            pred = pred2
            parse_reason = reason2
        else:
            return {
                "pred": UNKNOWN_LABEL,
                "reason": UNK_API_ERROR,
                "raw": f"__API_ERROR__ {str(err_msg2)[:600]}",
                "calls_made": calls_made,
                "err_calls": int(att2),
                "call_status": "api_error_after_invalid_label_retry",
            }

    if pred == UNKNOWN_LABEL:
        return {
            "pred": UNKNOWN_LABEL,
            "reason": parse_reason,
            "raw": raw[:600],
            "calls_made": calls_made,
            "err_calls": 0,
            "call_status": "parsed_unknown",
        }

    return {
        "pred": pred,
        "reason": UNK_OK,
        "raw": raw[:600],
        "calls_made": calls_made,
        "err_calls": 0,
        "call_status": "ok",
    }


# ======================================================
# TEST CALL
# ======================================================
def run_test_call(model_id: str, model_name: str, cfg: dict):
    print(f"\n[TEST] Testing {model_name} with 1 request...")

    limiter = AdaptiveRateLimiter(
        initial_rps=min(0.5, cfg["initial_rps"]),
        min_rps=cfg["min_rps"],
        max_rps=max(1.0, cfg["max_rps"]),
    )

    test_text = normalize_arabic("اليوم فرحت كثير لما نجحت")
    raw, err_flag, err_text, calls_made = call_model(
        model_id=model_id,
        model_name=model_name,
        text=test_text,
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_output_tokens"],
        request_spacing=cfg["request_spacing"],
        connect_timeout=cfg["connect_timeout"],
        read_timeout=cfg["read_timeout"],
        rate_limiter=limiter,
        use_system_msg=cfg["use_system_msg"],
        cfg_reasoning_effort=cfg.get("reasoning_effort"),
        sample_i="TEST",
        vote_i=0,
    )

    print(f"[TEST] err_flag: {err_flag} | attempts: {calls_made}")
    print(f"[TEST] err: {err_text}")
    print(f"[TEST] raw_head: {raw[:200] if raw else '<<EMPTY>>'}")

    if err_flag == 1:
        raise RuntimeError(f"Test call failed for {model_name}: {err_text}")

    pred, reason = parse_label_with_reason(raw)
    print(f"[TEST] parsed: {pred} | reason: {reason}")

    if pred == UNKNOWN_LABEL:
        raise RuntimeError(f"Test call parsing failed for {model_name}. Raw: {raw[:400]}")


# ======================================================
# VOTING
# ======================================================
def _vote_pattern(counter: Counter) -> str:
    if not counter:
        return "0"
    return "-".join(map(str, sorted(counter.values(), reverse=True)))


def classify_one(
    model_id: str,
    model_name: str,
    text: str,
    cfg: dict,
    rate_limiter: AdaptiveRateLimiter,
    sample_i=None,
):
    vote_preds = []
    vote_raws = []
    vote_reasons = []
    vote_call_statuses = []

    calls_made_total = 0
    err_calls_total_local = 0

    for v in range(VOTES):
        one = resolve_single_vote(
            model_id=model_id,
            model_name=model_name,
            text=text,
            cfg=cfg,
            rate_limiter=rate_limiter,
            sample_suffix=f"{sample_i}__VOTE{v+1}",
        )
        vote_preds.append(one["pred"])
        vote_raws.append(one["raw"])
        vote_reasons.append(one["reason"])
        vote_call_statuses.append(one["call_status"])
        calls_made_total += int(one["calls_made"])
        err_calls_total_local += int(one["err_calls"])

    counts_all = Counter(vote_preds)
    unknown_votes = counts_all.get(UNKNOWN_LABEL, 0)

    valid_votes = [p for p in vote_preds if p in ALLOWED_LABELS]
    counts_valid = Counter(valid_votes)
    pattern = _vote_pattern(counts_valid)

    if len(valid_votes) == 0:
        if all(cs.startswith("api_error") for cs in vote_call_statuses):
            final_reason = FINAL_ALL_FAILED
        else:
            final_reason = FINAL_ALL_UNPARSEABLE

        return {
            "pred": UNKNOWN_LABEL,
            "err_calls": err_calls_total_local,
            "abstained": 1,
            "calls_made": calls_made_total,
            "vote_pattern": pattern,
            "unknown_votes": unknown_votes,
            "unknown_reason": final_reason,
            "votes_list": vote_preds,
            "raw_votes": vote_raws,
            "parse_reasons": vote_reasons,
            "call_statuses": vote_call_statuses,
        }

    if len(valid_votes) == VOTES and len(counts_valid) == VOTES:
        return {
            "pred": UNKNOWN_LABEL,
            "err_calls": err_calls_total_local,
            "abstained": 1,
            "calls_made": calls_made_total,
            "vote_pattern": pattern,
            "unknown_votes": unknown_votes,
            "unknown_reason": FINAL_ALL_DIFFERENT_111,
            "votes_list": vote_preds,
            "raw_votes": vote_raws,
            "parse_reasons": vote_reasons,
            "call_statuses": vote_call_statuses,
        }

    top_label, top_count = counts_valid.most_common(1)[0]
    if top_count >= 2:
        return {
            "pred": top_label,
            "err_calls": err_calls_total_local,
            "abstained": 0,
            "calls_made": calls_made_total,
            "vote_pattern": pattern,
            "unknown_votes": unknown_votes,
            "unknown_reason": FINAL_MAJORITY,
            "votes_list": vote_preds,
            "raw_votes": vote_raws,
            "parse_reasons": vote_reasons,
            "call_statuses": vote_call_statuses,
        }

    return {
        "pred": UNKNOWN_LABEL,
        "err_calls": err_calls_total_local,
        "abstained": 1,
        "calls_made": calls_made_total,
        "vote_pattern": pattern,
        "unknown_votes": unknown_votes,
        "unknown_reason": FINAL_NO_MAJORITY_11U,
        "votes_list": vote_preds,
        "raw_votes": vote_raws,
        "parse_reasons": vote_reasons,
        "call_statuses": vote_call_statuses,
    }


# ======================================================
# SAVE / RESUME HELPERS
# ======================================================
def get_prediction_path(model_name: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return os.path.join(PRED_DIR, f"{PROVIDER_NAME}__{safe_model}__LATEST_PREDICTIONS.csv")


def get_prediction_archive_path(model_name: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return os.path.join(PRED_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{safe_model}__PREDICTIONS.csv")


def save_partial_predictions(
    pred_path,
    model_name,
    preds,
    abstained_arr,
    vote_patterns,
    unknown_votes_arr,
    unknown_reasons,
    calls_made_arr,
    votes_list_arr,
    raw_vote_1_arr,
    raw_vote_2_arr,
    raw_vote_3_arr,
    parse_reason_1_arr,
    parse_reason_2_arr,
    parse_reason_3_arr,
    call_status_1_arr,
    call_status_2_arr,
    call_status_3_arr,
    cfg,
):
    out_df = pd.DataFrame({
        "row_id": list(range(len(df))),
        "text": df["text"],
        "gold_label": df["emotion"],
        "prediction": preds,
        "abstained": abstained_arr,
        "vote_pattern": vote_patterns,
        "unknown_votes": unknown_votes_arr,
        "unknown_reason": unknown_reasons,
        "calls_made": calls_made_arr,
        "provider": PROVIDER_NAME,
        "model": model_name,
        "run_id": RUN_ID,
        "votes": VOTES,
        "temp_votes": cfg["temperature"],
        "max_workers": cfg["max_workers"],
        "initial_rps": cfg["initial_rps"],
        "min_rps": cfg["min_rps"],
        "max_rps": cfg["max_rps"],
        "max_output_tokens": cfg["max_output_tokens"],
        "num_api_keys": len(CLIENTS),
    })

    if STORE_VOTES_LIST:
        out_df["votes_list"] = votes_list_arr

    if STORE_RAW_VOTES:
        out_df["raw_vote_1"] = raw_vote_1_arr
        out_df["raw_vote_2"] = raw_vote_2_arr
        out_df["raw_vote_3"] = raw_vote_3_arr
        out_df["parse_reason_1"] = parse_reason_1_arr
        out_df["parse_reason_2"] = parse_reason_2_arr
        out_df["parse_reason_3"] = parse_reason_3_arr
        out_df["call_status_1"] = call_status_1_arr
        out_df["call_status_2"] = call_status_2_arr
        out_df["call_status_3"] = call_status_3_arr

    out_df.to_csv(pred_path, index=False, encoding="utf-8-sig")


# ======================================================
# MAIN LOOP
# ======================================================
summary_rows = []

for MODEL_NAME, BASE_CFG in MODEL_CONFIGS.items():
    CFG = build_runtime_config(BASE_CFG, NUM_KEYS)
    MODEL_ID = CFG["model_id"]

    print(f"\nRunning {MODEL_NAME} (VOTES={VOTES})...")
    print(
        "Runtime config: "
        f"workers={CFG['max_workers']}, "
        f"initial_rps={CFG['initial_rps']:.2f}, "
        f"min_rps={CFG['min_rps']:.2f}, "
        f"max_rps={CFG['max_rps']:.2f}, "
        f"temperature={CFG['temperature']:.2f}, "
        f"request_spacing={CFG['request_spacing']}, "
        f"connect_timeout={CFG['connect_timeout']}, "
        f"read_timeout={CFG['read_timeout']}, "
        f"max_output_tokens={CFG['max_output_tokens']}, "
        f"keys={NUM_KEYS}"
    )

    if TEST_CALL_BEFORE_RUN:
        run_test_call(MODEL_ID, MODEL_NAME, CFG)

    rate_limiter = AdaptiveRateLimiter(
        initial_rps=CFG["initial_rps"],
        min_rps=CFG["min_rps"],
        max_rps=CFG["max_rps"],
    )

    n = len(df)
    pred_path = get_prediction_path(MODEL_NAME)

    preds = [None] * n
    abstained_arr = [None] * n
    vote_patterns = [None] * n
    unknown_votes_arr = [None] * n
    unknown_reasons = [None] * n
    calls_made_arr = [None] * n
    votes_list_arr = [None] * n

    raw_vote_1_arr = [None] * n
    raw_vote_2_arr = [None] * n
    raw_vote_3_arr = [None] * n

    parse_reason_1_arr = [None] * n
    parse_reason_2_arr = [None] * n
    parse_reason_3_arr = [None] * n

    call_status_1_arr = [None] * n
    call_status_2_arr = [None] * n
    call_status_3_arr = [None] * n

    resumed_done = 0

    if ENABLE_RESUME and os.path.exists(pred_path):
        old = pd.read_csv(pred_path, encoding="utf-8")
        if "row_id" in old.columns:
            for _, r in old.iterrows():
                i = int(r["row_id"])
                pred_val = r.get("prediction", None)

                if pd.notna(pred_val):
                    preds[i] = str(pred_val)
                    abstained_arr[i] = int(r["abstained"]) if pd.notna(r.get("abstained")) else 0
                    vote_patterns[i] = str(r["vote_pattern"]) if pd.notna(r.get("vote_pattern")) else None
                    unknown_votes_arr[i] = int(r["unknown_votes"]) if pd.notna(r.get("unknown_votes")) else 0
                    unknown_reasons[i] = str(r["unknown_reason"]) if pd.notna(r.get("unknown_reason")) else None
                    calls_made_arr[i] = int(r["calls_made"]) if pd.notna(r.get("calls_made")) else 0

                    if STORE_VOTES_LIST and "votes_list" in old.columns and pd.notna(r.get("votes_list")):
                        votes_list_arr[i] = r["votes_list"]

                    if STORE_RAW_VOTES:
                        for col_name, arr_ref in [
                            ("raw_vote_1", raw_vote_1_arr),
                            ("raw_vote_2", raw_vote_2_arr),
                            ("raw_vote_3", raw_vote_3_arr),
                            ("parse_reason_1", parse_reason_1_arr),
                            ("parse_reason_2", parse_reason_2_arr),
                            ("parse_reason_3", parse_reason_3_arr),
                            ("call_status_1", call_status_1_arr),
                            ("call_status_2", call_status_2_arr),
                            ("call_status_3", call_status_3_arr),
                        ]:
                            if col_name in old.columns and pd.notna(r.get(col_name)):
                                arr_ref[i] = r.get(col_name)

                    resumed_done += 1

        print(f"Resume detected for {MODEL_NAME}: loaded {resumed_done} completed rows.")

    remaining_indices = [i for i in range(n) if preds[i] is None]
    print(f"Remaining rows to run for {MODEL_NAME}: {len(remaining_indices)} / {n}")

    error_calls_total = 0
    abstain_count = sum(1 for x in abstained_arr if x == 1)

    debug_buffer = {}
    next_debug_to_print = 0

    if remaining_indices:
        with ThreadPoolExecutor(max_workers=CFG["max_workers"]) as executor:
            futures = {
                executor.submit(
                    classify_one,
                    MODEL_ID,
                    MODEL_NAME,
                    df.loc[i, "text"],
                    CFG,
                    rate_limiter,
                    i,
                ): i
                for i in remaining_indices
            }

            completed_since_save = 0

            for fut in tqdm(as_completed(futures), total=len(remaining_indices), desc=MODEL_NAME):
                i = futures[fut]

                try:
                    row = fut.result()
                except Exception as e:
                    row = {
                        "pred": UNKNOWN_LABEL,
                        "err_calls": VOTES,
                        "abstained": 1,
                        "calls_made": VOTES,
                        "vote_pattern": "0",
                        "unknown_votes": VOTES,
                        "unknown_reason": f"future_exception:{type(e).__name__}",
                        "votes_list": [UNKNOWN_LABEL] * VOTES,
                        "raw_votes": ["", "", ""],
                        "parse_reasons": ["future_exception"] * VOTES,
                        "call_statuses": [f"future_exception:{type(e).__name__}"] * VOTES,
                    }

                preds[i] = row["pred"]
                abstained_arr[i] = row["abstained"]
                vote_patterns[i] = row["vote_pattern"]
                unknown_votes_arr[i] = int(row["unknown_votes"])
                unknown_reasons[i] = row["unknown_reason"]
                calls_made_arr[i] = int(row["calls_made"])

                if STORE_VOTES_LIST:
                    votes_list_arr[i] = json.dumps(row["votes_list"], ensure_ascii=False)

                raw_votes = row["raw_votes"]
                parse_reasons = row["parse_reasons"]
                call_statuses = row["call_statuses"]

                if len(raw_votes) >= 1:
                    raw_vote_1_arr[i] = raw_votes[0]
                if len(raw_votes) >= 2:
                    raw_vote_2_arr[i] = raw_votes[1]
                if len(raw_votes) >= 3:
                    raw_vote_3_arr[i] = raw_votes[2]

                if len(parse_reasons) >= 1:
                    parse_reason_1_arr[i] = parse_reasons[0]
                if len(parse_reasons) >= 2:
                    parse_reason_2_arr[i] = parse_reasons[1]
                if len(parse_reasons) >= 3:
                    parse_reason_3_arr[i] = parse_reasons[2]

                if len(call_statuses) >= 1:
                    call_status_1_arr[i] = call_statuses[0]
                if len(call_statuses) >= 2:
                    call_status_2_arr[i] = call_statuses[1]
                if len(call_statuses) >= 3:
                    call_status_3_arr[i] = call_statuses[2]

                if i < DEBUG_PRINT_FIRST:
                    debug_buffer[i] = {
                        "gold": df.loc[i, "emotion"],
                        "pred": row["pred"],
                        "abst": row["abstained"],
                        "reason": row["unknown_reason"],
                        "calls": row["calls_made"],
                        "text": df.loc[i, "text"],
                        "votes": row["votes_list"],
                        "raw_votes": raw_votes,
                        "parse_reasons": parse_reasons,
                        "call_statuses": call_statuses,
                    }

                    while next_debug_to_print in debug_buffer:
                        item = debug_buffer.pop(next_debug_to_print)
                        print("-----")
                        print(
                            f"[{next_debug_to_print:04d}] "
                            f"GOLD={item['gold']}  "
                            f"PRED={item['pred']}  "
                            f"abst={item['abst']}  "
                            f"reason={item['reason']}  "
                            f"calls={item['calls']}"
                        )
                        print("TEXT :", item["text"])
                        print("VOTES:", item["votes"])
                        print("RAW1 :", repr(item["raw_votes"][0]) if len(item["raw_votes"]) > 0 else "")
                        print("RAW2 :", repr(item["raw_votes"][1]) if len(item["raw_votes"]) > 1 else "")
                        print("RAW3 :", repr(item["raw_votes"][2]) if len(item["raw_votes"]) > 2 else "")
                        print("PARSE:", item["parse_reasons"])
                        print("CALLS:", item["call_statuses"])
                        next_debug_to_print += 1

                error_calls_total += int(row["err_calls"])
                abstain_count += int(row["abstained"])
                completed_since_save += 1

                if completed_since_save >= SAVE_EVERY_N:
                    save_partial_predictions(
                        pred_path,
                        MODEL_NAME,
                        preds,
                        abstained_arr,
                        vote_patterns,
                        unknown_votes_arr,
                        unknown_reasons,
                        calls_made_arr,
                        votes_list_arr,
                        raw_vote_1_arr,
                        raw_vote_2_arr,
                        raw_vote_3_arr,
                        parse_reason_1_arr,
                        parse_reason_2_arr,
                        parse_reason_3_arr,
                        call_status_1_arr,
                        call_status_2_arr,
                        call_status_3_arr,
                        CFG,
                    )
                    completed_since_save = 0

        save_partial_predictions(
            pred_path,
            MODEL_NAME,
            preds,
            abstained_arr,
            vote_patterns,
            unknown_votes_arr,
            unknown_reasons,
            calls_made_arr,
            votes_list_arr,
            raw_vote_1_arr,
            raw_vote_2_arr,
            raw_vote_3_arr,
            parse_reason_1_arr,
            parse_reason_2_arr,
            parse_reason_3_arr,
            call_status_1_arr,
            call_status_2_arr,
            call_status_3_arr,
            CFG,
        )
    else:
        print(f"{MODEL_NAME}: nothing left to run. Using resumed predictions only.")

    if any(p is None for p in preds):
        missing_count = sum(1 for p in preds if p is None)
        raise RuntimeError(f"{MODEL_NAME}: still has {missing_count} missing predictions after run/resume.")

    y_true = df["emotion"].tolist()
    y_pred = preds

    unknown_total = y_pred.count(UNKNOWN_LABEL)
    unknown_rate = unknown_total / len(y_pred)
    coverage = 1.0 - unknown_rate

    pd.Series(unknown_reasons).value_counts(dropna=False).to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__UNKNOWN_REASON_DISTRIBUTION.csv"),
        encoding="utf-8-sig"
    )

    y_pred_strict = [p if p != UNKNOWN_LABEL else STRICT_UNKNOWN_SENTINEL for p in y_pred]

    strict_accuracy = accuracy_score(y_true, y_pred_strict)
    strict_macro_f1 = f1_score(
        y_true, y_pred_strict,
        average="macro",
        labels=ALLOWED_LABELS,
        zero_division=0
    )
    strict_micro_f1 = f1_score(
        y_true, y_pred_strict,
        average="micro",
        labels=ALLOWED_LABELS,
        zero_division=0
    )
    strict_weighted_f1 = f1_score(
        y_true, y_pred_strict,
        average="weighted",
        labels=ALLOWED_LABELS,
        zero_division=0
    )

    report_strict_5 = classification_report(
        y_true,
        y_pred_strict,
        labels=ALLOWED_LABELS,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report_strict_5).transpose().to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__REPORT_STRICT_5LABELS.csv"),
        encoding="utf-8-sig"
    )

    report_strict_with_unknown = classification_report(
        y_true,
        y_pred,
        labels=STRICT_LABELS,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report_strict_with_unknown).transpose().to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__REPORT_STRICT_WITH_UNKNOWN.csv"),
        encoding="utf-8-sig"
    )

    cm_strict_unknown = confusion_matrix(y_true, y_pred, labels=STRICT_LABELS)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm_strict_unknown, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(STRICT_LABELS)), STRICT_LABELS, rotation=45)
    plt.yticks(range(len(STRICT_LABELS)), STRICT_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{MODEL_NAME} Confusion Matrix (STRICT incl. unknown)")
    for i_cm in range(len(STRICT_LABELS)):
        for j_cm in range(len(STRICT_LABELS)):
            plt.text(j_cm, i_cm, str(cm_strict_unknown[i_cm, j_cm]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_STRICT_WITH_UNKNOWN.png"),
        dpi=300
    )
    plt.close()

    cm_strict_5 = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_strict_5, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(ALLOWED_LABELS)), ALLOWED_LABELS, rotation=45)
    plt.yticks(range(len(ALLOWED_LABELS)), ALLOWED_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{MODEL_NAME} Confusion Matrix (STRICT 5 labels)")
    for i_cm in range(len(ALLOWED_LABELS)):
        for j_cm in range(len(ALLOWED_LABELS)):
            plt.text(j_cm, i_cm, str(cm_strict_5[i_cm, j_cm]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_STRICT_5LABELS.png"),
        dpi=300
    )
    plt.close()

    mask = [p != UNKNOWN_LABEL for p in y_pred]
    y_true_cov = [yt for yt, m in zip(y_true, mask) if m]
    y_pred_cov = [yp for yp in y_pred if yp != UNKNOWN_LABEL]

    if len(y_true_cov) > 0:
        covered_accuracy = accuracy_score(y_true_cov, y_pred_cov)
        covered_macro_f1 = f1_score(
            y_true_cov, y_pred_cov,
            average="macro",
            labels=COVERED_LABELS,
            zero_division=0
        )
        covered_micro_f1 = f1_score(
            y_true_cov, y_pred_cov,
            average="micro",
            labels=COVERED_LABELS,
            zero_division=0
        )
        covered_weighted_f1 = f1_score(
            y_true_cov, y_pred_cov,
            average="weighted",
            labels=COVERED_LABELS,
            zero_division=0
        )
    else:
        covered_accuracy = 0.0
        covered_macro_f1 = 0.0
        covered_micro_f1 = 0.0
        covered_weighted_f1 = 0.0

    archive_pred_path = get_prediction_archive_path(MODEL_NAME)
    save_partial_predictions(
        archive_pred_path,
        MODEL_NAME,
        preds,
        abstained_arr,
        vote_patterns,
        unknown_votes_arr,
        unknown_reasons,
        calls_made_arr,
        votes_list_arr,
        raw_vote_1_arr,
        raw_vote_2_arr,
        raw_vote_3_arr,
        parse_reason_1_arr,
        parse_reason_2_arr,
        parse_reason_3_arr,
        call_status_1_arr,
        call_status_2_arr,
        call_status_3_arr,
        CFG,
    )

    reason_counter = Counter(unknown_reasons)

    summary_rows.append({
        "provider": PROVIDER_NAME,
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "setup": "zero-shot + three-vote self-consistency (T=0.3) + multi-key + strict-json + robust-parser + retries + resume + autosave",
        "votes": VOTES,
        "temp_votes": CFG["temperature"],
        "max_workers": CFG["max_workers"],
        "initial_rps": CFG["initial_rps"],
        "min_rps": CFG["min_rps"],
        "max_rps": CFG["max_rps"],
        "final_rps_estimate": rate_limiter.get_rps(),
        "num_api_keys": len(CLIENTS),
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

        "covered_accuracy": covered_accuracy,
        "covered_macro_f1": covered_macro_f1,
        "covered_micro_f1": covered_micro_f1,
        "covered_weighted_f1": covered_weighted_f1,

        "unknown_reason_majority": int(reason_counter.get(FINAL_MAJORITY, 0)),
        "unknown_reason_all_failed": int(reason_counter.get(FINAL_ALL_FAILED, 0)),
        "unknown_reason_all_unparseable": int(reason_counter.get(FINAL_ALL_UNPARSEABLE, 0)),
        "unknown_reason_all_different_111": int(reason_counter.get(FINAL_ALL_DIFFERENT_111, 0)),
        "unknown_reason_no_majority_11u": int(reason_counter.get(FINAL_NO_MAJORITY_11U, 0)),
    })


# ======================================================
# SAVE SUMMARY
# ======================================================
summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_Q1_hybrid_v2.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

latest_summary_path = os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__LATEST__SUMMARY_Q1_hybrid_v2.csv")
summary_df.to_csv(latest_summary_path, index=False, encoding="utf-8-sig")

print("\nGroq three-vote benchmark completed successfully for all manuscript Groq models.")
print("RUN_ID:", RUN_ID)
print("Loaded keys:", len(CLIENTS))
print("Models run:", list(MODEL_CONFIGS.keys()))
print("Fixed temperature:", FIXED_TEMPERATURE)
print("Predictions:", PRED_DIR)
print("Figures    :", FIG_DIR)
print("Metrics    :", METRIC_DIR)
print("Summary    :", summary_path)