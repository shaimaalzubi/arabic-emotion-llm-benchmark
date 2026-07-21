from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


# =============================================================================
# CONFIG — GOOGLE THREE-VOTE SELF-CONSISTENCY
# =============================================================================
VOTES = 3
FIXED_TEMPERATURE = 0.3

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_LABELS = ALLOWED_LABELS + [UNKNOWN_LABEL]
COVERED_LABELS = ALLOWED_LABELS
STRICT_UNKNOWN_SENTINEL = "__unknown__"

PROVIDER_NAME = "Google"

MAX_RETRIES = 4
BASE_BACKOFF = 1.2
JITTER = 0.40

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

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__THREE_VOTE"


# =============================================================================
# UNKNOWN / FINAL REASONS
# =============================================================================
UNK_OK = "ok"
UNK_API_ERROR = "api_error"
UNK_HTTP_STATUS = "http_status"
UNK_EMPTY_RESPONSE = "empty_response"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"
UNK_REGEX_LABEL = "regex_label"
UNK_REPAIRED_TRUNC = "repaired_truncation"
UNK_NON_JSON_PREFACE = "non_json_preface"
UNK_PROMPT_BLOCKED = "prompt_blocked"
UNK_MAX_TOKENS = "max_tokens"
UNK_REQUEST_TIMEOUT = "request_timeout"
UNK_EXCEPTION = "exception"

FINAL_MAJORITY = "majority"
FINAL_ALL_FAILED = "all_failed"
FINAL_ALL_UNPARSEABLE = "all_unparseable"
FINAL_ALL_DIFFERENT_111 = "all_different_111"
FINAL_NO_MAJORITY_11U = "no_majority_11u"


# =============================================================================
# MODEL CONFIGS
# =============================================================================
# Gemini Flash uses Google structured output and a small thinking budget.
# Gemma is kept on prompt + parser validation for broader compatibility.
MODEL_CONFIGS = {
    "Gemini-2.5-Flash": {
        "model_id": "models/gemini-2.5-flash",
        "max_workers": 2,
        "initial_rps": 0.35,
        "min_rps": 0.10,
        "max_rps": 0.60,
        "request_spacing": 0.00,
        "connect_timeout": 20,
        "read_timeout": 60,
        "max_output_tokens": 512,
        "use_structured_output": True,
        "thinking_budget": 128,
    },
    "Gemma-3-12B": {
        # Historical model identifier used for the reported benchmark.
        # Availability may depend on account, region, and current API catalogue.
        "model_id": "models/gemma-3-12b-it",
        "max_workers": 2,
        "initial_rps": 0.25,
        "min_rps": 0.08,
        "max_rps": 0.50,
        "request_spacing": 0.02,
        "connect_timeout": 20,
        "read_timeout": 60,
        "max_output_tokens": 128,
        "use_structured_output": False,
        "thinking_budget": None,
    },
}


# =============================================================================
# API / SESSION
# =============================================================================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment (.env).")

SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=50,
    pool_maxsize=50,
    max_retries=0,
)
SESSION.mount("https://", ADAPTER)
SESSION.headers.update({"Content-Type": "application/json"})


# =============================================================================
# PATHS
# =============================================================================
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


# =============================================================================
# DATA
# =============================================================================
df = pd.read_csv(DATA_PATH, encoding="utf-8")

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


def normalize_arabic(text: str) -> str:
    text = str(text)
    text = re.sub(r"[إأٱآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["text"] = df["text"].apply(normalize_arabic)

df["emotion"].value_counts().to_csv(
    os.path.join(
        METRIC_DIR,
        f"{PROVIDER_NAME}__RUN{RUN_ID}__DATASET_class_distribution.csv",
    ),
    encoding="utf-8-sig",
)


# =============================================================================
# PROMPTS
# =============================================================================
SYSTEM_MSG = (
    "You are a strict emotion classifier for Arabic text. "
    "Return only one valid JSON object with exactly one key named label. "
    "The label must be one of: joy, anger, sadness, fear, neutral. "
    "Use lowercase English labels only. "
    "Do not explain. Do not output markdown or extra text."
)

SYSTEM_MSG_RETRY = (
    'Return only one complete JSON object exactly like {"label":"neutral"}. '
    "Allowed labels only: joy, anger, sadness, fear, neutral."
)

SYSTEM_MSG_INVALID_LABEL_RETRY = (
    'Your previous label was invalid. Return only JSON like {"label":"joy"} '
    "using one of: joy, anger, sadness, fear, neutral."
)

SYSTEM_MSG_PARSE_FAIL_RETRY = (
    'Your previous output was invalid or truncated. Return only one complete '
    'JSON object like {"label":"neutral"}.'
)

PROMPT_BASE = """
Classify the emotion of the following Arabic sentence.

Rules:
- Choose exactly one label from: joy, anger, sadness, fear, neutral
- Return only JSON: {"label":"joy"}
- Use lowercase English labels only
- Do not explain
- If unsure, choose the single best allowed label
- Never output unknown

Sentence:
"{TEXT}"
""".strip()


# =============================================================================
# RATE LIMITER
# =============================================================================
class AdaptiveRateLimiter:
    def __init__(self, initial_rps: float, min_rps: float, max_rps: float):
        self.current_rps = max(min(initial_rps, max_rps), min_rps)
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.lock = threading.Lock()
        self.next_allowed_time = 0.0
        self.consecutive_429 = 0
        self.cooldown_until = 0.0

    def wait_for_slot(self):
        while True:
            with self.lock:
                now = time.time()

                if now < self.cooldown_until:
                    sleep_for = self.cooldown_until - now
                elif now < self.next_allowed_time:
                    sleep_for = self.next_allowed_time - now
                else:
                    interval = 1.0 / max(self.current_rps, 1e-6)
                    self.next_allowed_time = now + interval
                    return

            time.sleep(max(0.0, sleep_for))

    def on_success(self):
        with self.lock:
            self.consecutive_429 = max(0, self.consecutive_429 - 1)
            self.current_rps = min(
                self.max_rps,
                self.current_rps * 1.05 + 0.003,
            )

    def on_429(self, retry_after: Optional[float] = None):
        with self.lock:
            self.consecutive_429 += 1
            self.current_rps = max(
                self.min_rps,
                self.current_rps * (0.75 if self.consecutive_429 == 1 else 0.60),
            )
            cooldown = (
                float(retry_after)
                if retry_after is not None
                else min(30.0, 3.0 * self.consecutive_429)
            )
            self.cooldown_until = max(
                self.cooldown_until,
                time.time() + max(0.0, cooldown),
            )

    def on_transient_error(self):
        with self.lock:
            self.current_rps = max(
                self.min_rps,
                self.current_rps * 0.75,
            )

    def get_rps(self) -> float:
        with self.lock:
            return self.current_rps


# =============================================================================
# PARSER
# =============================================================================
def _strip_code_fences(text: str) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"```[^\n]*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"```", "", value)
    return value.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    value = _strip_code_fences(text)
    if value.startswith("{") and value.endswith("}"):
        return value

    match = re.search(r"\{.*?\}", value, flags=re.DOTALL)
    return match.group(0).strip() if match else None


def _try_load_json_obj(blob: str):
    for candidate in (blob, blob.replace("'", '"')):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def clean_label(value: Any) -> str:
    if value is None:
        return UNKNOWN_LABEL

    label = str(value).strip().lower()
    label = re.sub(r"[^a-z_\-\s]", " ", label)
    label = re.sub(r"\s+", " ", label).strip()

    if label in ALLOWED_LABELS:
        return label

    first = label.split(" ")[0] if label else ""
    if first in ALLOWED_LABELS:
        return first

    for allowed in ALLOWED_LABELS:
        if re.search(rf"\b{re.escape(allowed)}\b", label):
            return allowed

    return UNKNOWN_LABEL


def _regex_label_anywhere(raw: str) -> Optional[str]:
    if not raw:
        return None

    low = _strip_code_fences(raw).lower()

    match = re.search(
        r'\blabel\b\s*[:=]\s*["\']?\s*'
        r'(joy|anger|sadness|fear|neutral)\s*["\']?',
        low,
    )
    if match:
        return match.group(1)

    hits = [
        label
        for label in ALLOWED_LABELS
        if re.search(rf"\b{re.escape(label)}\b", low)
    ]
    return hits[0] if len(hits) == 1 else None


def repair_truncated_label_json(raw: str) -> Optional[str]:
    if not raw:
        return None

    value = raw.strip()
    if "{" in value and "}" in value:
        return None

    match = re.search(
        r'\{\s*"\s*label\s*"\s*:\s*"\s*([a-zA-Z]{1,20})\s*$',
        value,
    )
    if not match:
        return None

    prefix = match.group(1).lower()
    matches = [label for label in ALLOWED_LABELS if label.startswith(prefix)]
    if len(matches) == 1:
        return json.dumps({"label": matches[0]})
    return None


def parse_label_with_reason(raw_text: str):
    raw = str(raw_text).strip() if raw_text is not None else ""
    if not raw:
        return UNKNOWN_LABEL, UNK_EMPTY_RESPONSE

    blob = _extract_first_json_object(raw)
    if blob is not None:
        obj = _try_load_json_obj(blob)

        if not isinstance(obj, dict):
            fallback = _regex_label_anywhere(raw)
            if fallback:
                return fallback, UNK_REGEX_LABEL
            return UNKNOWN_LABEL, UNK_INVALID_JSON_PARSE_ERROR

        label_key = next(
            (key for key in obj if str(key).strip().lower() == "label"),
            None,
        )
        if label_key is None:
            fallback = _regex_label_anywhere(raw)
            if fallback:
                return fallback, UNK_REGEX_LABEL
            return UNKNOWN_LABEL, UNK_INVALID_JSON_MISSING_LABEL

        label = clean_label(obj.get(label_key))
        if label == UNKNOWN_LABEL:
            fallback = _regex_label_anywhere(raw)
            if fallback:
                return fallback, UNK_REGEX_LABEL
            return UNKNOWN_LABEL, UNK_INVALID_LABEL

        return label, UNK_OK

    repaired = repair_truncated_label_json(raw)
    if repaired:
        obj = _try_load_json_obj(repaired)
        if isinstance(obj, dict):
            label = clean_label(obj.get("label"))
            if label in ALLOWED_LABELS:
                return label, UNK_REPAIRED_TRUNC

    fallback = _regex_label_anywhere(raw)
    if fallback:
        return fallback, UNK_REGEX_LABEL

    if "{" not in raw:
        return UNKNOWN_LABEL, UNK_NON_JSON_PREFACE

    return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT


# =============================================================================
# BACKOFF
# =============================================================================
def _det_jitter(seed: int, attempt: int, salt: str) -> float:
    key = f"{seed}|{attempt}|{salt}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    fraction = int(digest[:8], 16) / float(16**8)
    return fraction * JITTER


def _sleep_backoff(attempt: int, salt: str):
    base = BASE_BACKOFF * (2 ** (attempt - 1))
    jitter = (
        _det_jitter(SEED, attempt, salt)
        if DETERMINISTIC_JITTER
        else random.uniform(0, JITTER)
    )
    time.sleep(base + jitter)


def _retry_after_seconds(response: requests.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if value is None:
        return None

    try:
        return float(str(value).strip().replace("s", ""))
    except Exception:
        return None


# =============================================================================
# GOOGLE RESPONSE HELPERS
# =============================================================================
def extract_candidate_text(data: dict) -> str:
    chunks: list[str] = []

    for candidate in data.get("candidates", []) or []:
        content = (candidate or {}).get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict):
                text = part.get("text")
                if text is not None and str(text).strip():
                    chunks.append(str(text))

    return "".join(chunks).strip()


def inspect_response(data: dict) -> dict[str, Any]:
    feedback = data.get("promptFeedback", {}) or {}

    finish_reasons = []
    finish_messages = []

    for candidate in data.get("candidates", []) or []:
        if isinstance(candidate, dict):
            if candidate.get("finishReason") is not None:
                finish_reasons.append(str(candidate["finishReason"]))
            if candidate.get("finishMessage") is not None:
                finish_messages.append(str(candidate["finishMessage"]))

    return {
        "block_reason": feedback.get("blockReason"),
        "finish_reasons": finish_reasons,
        "finish_messages": finish_messages,
        "usage": data.get("usageMetadata", {}) or {},
    }


# =============================================================================
# REQUEST BUILDING
# =============================================================================
def build_generation_config(model_cfg: dict, temperature: float) -> dict:
    generation_config: dict[str, Any] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(model_cfg["max_output_tokens"]),
        "candidateCount": 1,
    }

    if model_cfg.get("use_structured_output", False):
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = {
            "type": "OBJECT",
            "properties": {
                "label": {
                    "type": "STRING",
                    "enum": ALLOWED_LABELS,
                }
            },
            "required": ["label"],
        }

    thinking_budget = model_cfg.get("thinking_budget")
    if thinking_budget is not None:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": int(thinking_budget)
        }

    return generation_config


def call_model(
    model_name: str,
    model_cfg: dict,
    text: str,
    temperature: float,
    rate_limiter: AdaptiveRateLimiter,
    sample_i=None,
    vote_i=None,
    force_system_msg: Optional[str] = None,
):
    model_id = model_cfg["model_id"]
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{model_id}:generateContent?key={API_KEY}"
    )

    prompt = PROMPT_BASE.replace("{TEXT}", text.replace('"', '\\"'))
    salt = f"{PROVIDER_NAME}|{model_name}|{sample_i}|{vote_i}"

    attempts_used = 0
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt
        system_message = (
            force_system_msg
            if force_system_msg is not None
            else (SYSTEM_MSG if attempt == 1 else SYSTEM_MSG_RETRY)
        )

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"{system_message}\n\n{prompt}"
                        }
                    ],
                }
            ],
            "generationConfig": build_generation_config(
                model_cfg,
                temperature,
            ),
        }

        try:
            rate_limiter.wait_for_slot()

            spacing = float(model_cfg.get("request_spacing", 0.0))
            if spacing > 0:
                time.sleep(spacing)

            response = SESSION.post(
                url,
                json=body,
                timeout=(
                    int(model_cfg["connect_timeout"]),
                    int(model_cfg["read_timeout"]),
                ),
            )

            if response.status_code != 200:
                last_error = (
                    f"{UNK_HTTP_STATUS}_{response.status_code}__"
                    f"{response.text[:500]}"
                )

                if response.status_code == 429:
                    rate_limiter.on_429(
                        retry_after=_retry_after_seconds(response)
                    )
                elif response.status_code in {408, 500, 502, 503, 504}:
                    rate_limiter.on_transient_error()

                if (
                    response.status_code in {408, 429, 500, 502, 503, 504}
                    and attempt < MAX_RETRIES
                ):
                    _sleep_backoff(attempt, salt)
                    continue

                return "", 1, last_error, attempts_used

            data = response.json()

            if "error" in data:
                error = data.get("error", {}) or {}
                code = error.get("code")
                message = str(error.get("message", ""))

                last_error = f"{UNK_API_ERROR}_{code}__{message[:500]}"

                if code == 429:
                    rate_limiter.on_429()
                elif code in {500, 502, 503, 504}:
                    rate_limiter.on_transient_error()

                if code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                    _sleep_backoff(attempt, salt)
                    continue

                return "", 1, last_error, attempts_used

            info = inspect_response(data)

            if info["block_reason"] is not None:
                return (
                    "",
                    1,
                    f"{UNK_PROMPT_BLOCKED}__{info['block_reason']}",
                    attempts_used,
                )

            raw = extract_candidate_text(data)

            if not raw:
                finish_reason = (
                    ",".join(info["finish_reasons"])
                    if info["finish_reasons"]
                    else "NO_FINISH_REASON"
                )
                finish_message = " | ".join(info["finish_messages"])

                reason = UNK_EMPTY_RESPONSE
                if "MAX_TOKENS" in finish_reason:
                    reason = UNK_MAX_TOKENS

                last_error = (
                    f"{reason}__finish={finish_reason}__msg={finish_message}"
                )

                rate_limiter.on_transient_error()

                if attempt < MAX_RETRIES:
                    _sleep_backoff(attempt, salt)
                    continue

                return "", 1, last_error, attempts_used

            rate_limiter.on_success()
            return raw.strip(), 0, None, attempts_used

        except requests.exceptions.Timeout as error:
            last_error = f"{UNK_REQUEST_TIMEOUT}__{repr(error)[:500]}"
            rate_limiter.on_transient_error()

            if attempt < MAX_RETRIES:
                _sleep_backoff(attempt, salt)
                continue

        except requests.exceptions.ConnectionError as error:
            last_error = (
                f"{UNK_EXCEPTION}__connection_error__{repr(error)[:500]}"
            )
            rate_limiter.on_transient_error()

            if attempt < MAX_RETRIES:
                _sleep_backoff(attempt, salt)
                continue

        except Exception as error:
            last_error = f"{UNK_EXCEPTION}__{repr(error)[:500]}"
            rate_limiter.on_transient_error()

            if attempt < MAX_RETRIES:
                _sleep_backoff(attempt, salt)
                continue

    return "", 1, (last_error or "max_retries_exceeded"), attempts_used


# =============================================================================
# SINGLE-VOTE RESOLUTION
# =============================================================================
def resolve_single_vote(
    model_name: str,
    model_cfg: dict,
    text: str,
    rate_limiter: AdaptiveRateLimiter,
    sample_suffix: str,
):
    raw, err_flag, err_text, attempts_used = call_model(
        model_name=model_name,
        model_cfg=model_cfg,
        text=text,
        temperature=FIXED_TEMPERATURE,
        rate_limiter=rate_limiter,
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

    if (
        RETRY_ON_PARSE_FAILURE
        and pred == UNKNOWN_LABEL
        and parse_reason in {
            UNK_INVALID_JSON_NO_OBJECT,
            UNK_INVALID_JSON_PARSE_ERROR,
            UNK_INVALID_JSON_MISSING_LABEL,
            UNK_NON_JSON_PREFACE,
        }
    ):
        raw2, err2, err_msg2, attempts2 = call_model(
            model_name=model_name,
            model_cfg=model_cfg,
            text=text,
            temperature=FIXED_TEMPERATURE,
            rate_limiter=rate_limiter,
            sample_i=f"{sample_suffix}__PARSE_RETRY",
            vote_i=0,
            force_system_msg=SYSTEM_MSG_PARSE_FAIL_RETRY,
        )
        calls_made += int(attempts2)

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
                "err_calls": int(attempts2),
                "call_status": "api_error_after_parse_retry",
            }

    if (
        pred == UNKNOWN_LABEL
        and parse_reason == UNK_INVALID_LABEL
        and RETRY_ON_INVALID_LABEL
    ):
        raw2, err2, err_msg2, attempts2 = call_model(
            model_name=model_name,
            model_cfg=model_cfg,
            text=text,
            temperature=FIXED_TEMPERATURE,
            rate_limiter=rate_limiter,
            sample_i=f"{sample_suffix}__INVLBL_RETRY",
            vote_i=0,
            force_system_msg=SYSTEM_MSG_INVALID_LABEL_RETRY,
        )
        calls_made += int(attempts2)

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
                "err_calls": int(attempts2),
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


# =============================================================================
# TEST CALL
# =============================================================================
def run_test_call(
    model_name: str,
    model_cfg: dict,
):
    print(f"\n[TEST] Testing {model_name} with 1 request...")

    limiter = AdaptiveRateLimiter(
        initial_rps=min(0.5, float(model_cfg["initial_rps"])),
        min_rps=float(model_cfg["min_rps"]),
        max_rps=max(1.0, float(model_cfg["max_rps"])),
    )

    raw, err_flag, err_text, calls_made = call_model(
        model_name=model_name,
        model_cfg=model_cfg,
        text=normalize_arabic("اليوم فرحت كثير لما نجحت"),
        temperature=FIXED_TEMPERATURE,
        rate_limiter=limiter,
        sample_i="TEST",
        vote_i=0,
    )

    print(f"[TEST] err_flag: {err_flag} | attempts: {calls_made}")
    print(f"[TEST] err: {err_text}")
    print(f"[TEST] raw_head: {raw[:200] if raw else '<<EMPTY>>'}")

    if err_flag == 1:
        raise RuntimeError(
            f"Test call failed for {model_name}: {err_text}"
        )

    pred, reason = parse_label_with_reason(raw)
    print(f"[TEST] parsed: {pred} | reason: {reason}")

    if pred == UNKNOWN_LABEL:
        raise RuntimeError(
            f"Test call parsing failed for {model_name}. Raw: {raw[:400]}"
        )


# =============================================================================
# THREE-VOTE AGGREGATION
# =============================================================================
def _vote_pattern(counter: Counter) -> str:
    if not counter:
        return "0"
    return "-".join(map(str, sorted(counter.values(), reverse=True)))


def classify_one(
    model_name: str,
    model_cfg: dict,
    text: str,
    rate_limiter: AdaptiveRateLimiter,
    sample_i=None,
):
    vote_preds = []
    vote_raws = []
    vote_reasons = []
    vote_call_statuses = []

    calls_made_total = 0
    err_calls_total = 0

    for vote_index in range(VOTES):
        result = resolve_single_vote(
            model_name=model_name,
            model_cfg=model_cfg,
            text=text,
            rate_limiter=rate_limiter,
            sample_suffix=f"{sample_i}__VOTE{vote_index + 1}",
        )

        vote_preds.append(result["pred"])
        vote_raws.append(result["raw"])
        vote_reasons.append(result["reason"])
        vote_call_statuses.append(result["call_status"])

        calls_made_total += int(result["calls_made"])
        err_calls_total += int(result["err_calls"])

    unknown_votes = vote_preds.count(UNKNOWN_LABEL)
    valid_votes = [
        prediction
        for prediction in vote_preds
        if prediction in ALLOWED_LABELS
    ]
    counts_valid = Counter(valid_votes)
    pattern = _vote_pattern(counts_valid)

    if not valid_votes:
        final_reason = (
            FINAL_ALL_FAILED
            if all(
                status.startswith("api_error")
                for status in vote_call_statuses
            )
            else FINAL_ALL_UNPARSEABLE
        )
        final_pred = UNKNOWN_LABEL
        abstained = 1

    elif len(valid_votes) == VOTES and len(counts_valid) == VOTES:
        final_pred = UNKNOWN_LABEL
        final_reason = FINAL_ALL_DIFFERENT_111
        abstained = 1

    else:
        top_label, top_count = counts_valid.most_common(1)[0]

        if top_count >= 2:
            final_pred = top_label
            final_reason = FINAL_MAJORITY
            abstained = 0
        else:
            final_pred = UNKNOWN_LABEL
            final_reason = FINAL_NO_MAJORITY_11U
            abstained = 1

    return {
        "pred": final_pred,
        "err_calls": err_calls_total,
        "abstained": abstained,
        "calls_made": calls_made_total,
        "vote_pattern": pattern,
        "unknown_votes": unknown_votes,
        "unknown_reason": final_reason,
        "votes_list": vote_preds,
        "raw_votes": vote_raws,
        "parse_reasons": vote_reasons,
        "call_statuses": vote_call_statuses,
    }


# =============================================================================
# SAVE / RESUME
# =============================================================================
def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


def get_prediction_path(model_name: str) -> str:
    return os.path.join(
        PRED_DIR,
        f"{PROVIDER_NAME}__{safe_model_name(model_name)}"
        f"__LATEST_THREE_VOTE_PREDICTIONS.csv",
    )


def get_prediction_archive_path(model_name: str) -> str:
    return os.path.join(
        PRED_DIR,
        f"{PROVIDER_NAME}__RUN{RUN_ID}__{safe_model_name(model_name)}"
        f"__THREE_VOTE_PREDICTIONS.csv",
    )


def save_partial_predictions(
    pred_path: str,
    model_name: str,
    model_cfg: dict,
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
):
    output = pd.DataFrame({
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
        "model_id": model_cfg["model_id"],
        "run_id": RUN_ID,
        "votes": VOTES,
        "temp_votes": FIXED_TEMPERATURE,
        "max_workers": model_cfg["max_workers"],
        "initial_rps": model_cfg["initial_rps"],
        "min_rps": model_cfg["min_rps"],
        "max_rps": model_cfg["max_rps"],
        "max_output_tokens": model_cfg["max_output_tokens"],
        "structured_output": model_cfg["use_structured_output"],
        "thinking_budget": model_cfg["thinking_budget"],
    })

    if STORE_VOTES_LIST:
        output["votes_list"] = votes_list_arr

    if STORE_RAW_VOTES:
        output["raw_vote_1"] = raw_vote_1_arr
        output["raw_vote_2"] = raw_vote_2_arr
        output["raw_vote_3"] = raw_vote_3_arr
        output["parse_reason_1"] = parse_reason_1_arr
        output["parse_reason_2"] = parse_reason_2_arr
        output["parse_reason_3"] = parse_reason_3_arr
        output["call_status_1"] = call_status_1_arr
        output["call_status_2"] = call_status_2_arr
        output["call_status_3"] = call_status_3_arr

    output.to_csv(
        pred_path,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# FIGURES
# =============================================================================
def save_confusion(
    y_true,
    y_pred,
    labels,
    title,
    path,
):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN LOOP
# =============================================================================
summary_rows = []

for MODEL_NAME, MODEL_CFG in MODEL_CONFIGS.items():
    print(f"\nRunning {MODEL_NAME} (VOTES={VOTES}, T={FIXED_TEMPERATURE})...")
    print(
        "Runtime config: "
        f"workers={MODEL_CFG['max_workers']}, "
        f"initial_rps={MODEL_CFG['initial_rps']}, "
        f"min_rps={MODEL_CFG['min_rps']}, "
        f"max_rps={MODEL_CFG['max_rps']}, "
        f"max_output_tokens={MODEL_CFG['max_output_tokens']}, "
        f"structured_output={MODEL_CFG['use_structured_output']}, "
        f"thinking_budget={MODEL_CFG['thinking_budget']}"
    )

    if TEST_CALL_BEFORE_RUN:
        run_test_call(MODEL_NAME, MODEL_CFG)

    rate_limiter = AdaptiveRateLimiter(
        initial_rps=float(MODEL_CFG["initial_rps"]),
        min_rps=float(MODEL_CFG["min_rps"]),
        max_rps=float(MODEL_CFG["max_rps"]),
    )

    n = len(df)
    prediction_path = get_prediction_path(MODEL_NAME)

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

    if ENABLE_RESUME and os.path.exists(prediction_path):
        old = pd.read_csv(prediction_path, encoding="utf-8")

        if "row_id" in old.columns:
            for _, row in old.iterrows():
                index = int(row["row_id"])
                prediction = row.get("prediction")

                if pd.notna(prediction):
                    preds[index] = str(prediction)
                    abstained_arr[index] = int(row.get("abstained", 0))
                    vote_patterns[index] = row.get("vote_pattern")
                    unknown_votes_arr[index] = int(
                        row.get("unknown_votes", 0)
                    )
                    unknown_reasons[index] = row.get("unknown_reason")
                    calls_made_arr[index] = int(row.get("calls_made", 0))

                    if (
                        STORE_VOTES_LIST
                        and "votes_list" in old.columns
                        and pd.notna(row.get("votes_list"))
                    ):
                        votes_list_arr[index] = row["votes_list"]

                    for column_name, target in [
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
                        if (
                            column_name in old.columns
                            and pd.notna(row.get(column_name))
                        ):
                            target[index] = row.get(column_name)

                    resumed_done += 1

        print(
            f"Resume detected for {MODEL_NAME}: "
            f"loaded {resumed_done} completed rows."
        )

    remaining_indices = [
        index
        for index in range(n)
        if preds[index] is None
    ]
    print(
        f"Remaining rows for {MODEL_NAME}: "
        f"{len(remaining_indices)} / {n}"
    )

    error_calls_total = 0
    abstain_count = sum(
        1
        for value in abstained_arr
        if value == 1
    )

    debug_buffer = {}
    next_debug_to_print = 0

    if remaining_indices:
        with ThreadPoolExecutor(
            max_workers=int(MODEL_CFG["max_workers"])
        ) as executor:
            futures = {
                executor.submit(
                    classify_one,
                    MODEL_NAME,
                    MODEL_CFG,
                    df.loc[index, "text"],
                    rate_limiter,
                    index,
                ): index
                for index in remaining_indices
            }

            completed_since_save = 0

            for future in tqdm(
                as_completed(futures),
                total=len(remaining_indices),
                desc=MODEL_NAME,
            ):
                index = futures[future]

                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "pred": UNKNOWN_LABEL,
                        "err_calls": VOTES,
                        "abstained": 1,
                        "calls_made": VOTES,
                        "vote_pattern": "0",
                        "unknown_votes": VOTES,
                        "unknown_reason": (
                            f"future_exception:{type(error).__name__}"
                        ),
                        "votes_list": [UNKNOWN_LABEL] * VOTES,
                        "raw_votes": [""] * VOTES,
                        "parse_reasons": ["future_exception"] * VOTES,
                        "call_statuses": [
                            f"future_exception:{type(error).__name__}"
                        ] * VOTES,
                    }

                preds[index] = result["pred"]
                abstained_arr[index] = result["abstained"]
                vote_patterns[index] = result["vote_pattern"]
                unknown_votes_arr[index] = int(result["unknown_votes"])
                unknown_reasons[index] = result["unknown_reason"]
                calls_made_arr[index] = int(result["calls_made"])

                if STORE_VOTES_LIST:
                    votes_list_arr[index] = json.dumps(
                        result["votes_list"],
                        ensure_ascii=False,
                    )

                raw_votes = result["raw_votes"]
                parse_reasons = result["parse_reasons"]
                call_statuses = result["call_statuses"]

                raw_vote_1_arr[index] = raw_votes[0]
                raw_vote_2_arr[index] = raw_votes[1]
                raw_vote_3_arr[index] = raw_votes[2]

                parse_reason_1_arr[index] = parse_reasons[0]
                parse_reason_2_arr[index] = parse_reasons[1]
                parse_reason_3_arr[index] = parse_reasons[2]

                call_status_1_arr[index] = call_statuses[0]
                call_status_2_arr[index] = call_statuses[1]
                call_status_3_arr[index] = call_statuses[2]

                if index < DEBUG_PRINT_FIRST:
                    debug_buffer[index] = {
                        "gold": df.loc[index, "emotion"],
                        "pred": result["pred"],
                        "abstained": result["abstained"],
                        "reason": result["unknown_reason"],
                        "calls": result["calls_made"],
                        "text": df.loc[index, "text"],
                        "votes": result["votes_list"],
                        "raw_votes": raw_votes,
                        "parse_reasons": parse_reasons,
                        "call_statuses": call_statuses,
                    }

                    while next_debug_to_print in debug_buffer:
                        item = debug_buffer.pop(next_debug_to_print)

                        print("-----")
                        print(
                            f"[{next_debug_to_print:04d}] "
                            f"GOLD={item['gold']} "
                            f"PRED={item['pred']} "
                            f"abst={item['abstained']} "
                            f"reason={item['reason']} "
                            f"calls={item['calls']}"
                        )
                        print("TEXT :", item["text"])
                        print("VOTES:", item["votes"])
                        print("RAW1 :", repr(item["raw_votes"][0]))
                        print("RAW2 :", repr(item["raw_votes"][1]))
                        print("RAW3 :", repr(item["raw_votes"][2]))
                        print("PARSE:", item["parse_reasons"])
                        print("CALLS:", item["call_statuses"])

                        next_debug_to_print += 1

                error_calls_total += int(result["err_calls"])
                abstain_count += int(result["abstained"])
                completed_since_save += 1

                if completed_since_save >= SAVE_EVERY_N:
                    save_partial_predictions(
                        prediction_path,
                        MODEL_NAME,
                        MODEL_CFG,
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
                    )
                    completed_since_save = 0

        save_partial_predictions(
            prediction_path,
            MODEL_NAME,
            MODEL_CFG,
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
        )

    if any(prediction is None for prediction in preds):
        missing_count = sum(
            prediction is None
            for prediction in preds
        )
        raise RuntimeError(
            f"{MODEL_NAME}: {missing_count} predictions are still missing."
        )

    y_true = df["emotion"].tolist()
    y_pred = preds

    unknown_total = y_pred.count(UNKNOWN_LABEL)
    unknown_rate = unknown_total / len(y_pred)
    coverage = 1.0 - unknown_rate

    pd.Series(unknown_reasons).value_counts(dropna=False).to_csv(
        os.path.join(
            METRIC_DIR,
            f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}"
            f"__UNKNOWN_REASON_DISTRIBUTION.csv",
        ),
        encoding="utf-8-sig",
    )

    strict_predictions = [
        prediction
        if prediction != UNKNOWN_LABEL
        else STRICT_UNKNOWN_SENTINEL
        for prediction in y_pred
    ]

    strict_accuracy = accuracy_score(y_true, strict_predictions)
    strict_macro_f1 = f1_score(
        y_true,
        strict_predictions,
        labels=ALLOWED_LABELS,
        average="macro",
        zero_division=0,
    )
    strict_micro_f1 = f1_score(
        y_true,
        strict_predictions,
        labels=ALLOWED_LABELS,
        average="micro",
        zero_division=0,
    )
    strict_weighted_f1 = f1_score(
        y_true,
        strict_predictions,
        labels=ALLOWED_LABELS,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        strict_predictions,
        labels=ALLOWED_LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(
            METRIC_DIR,
            f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}"
            f"__REPORT_STRICT_5LABELS.csv",
        ),
        encoding="utf-8-sig",
    )

    save_confusion(
        y_true,
        y_pred,
        STRICT_LABELS,
        f"{MODEL_NAME} Confusion Matrix (STRICT incl. unknown)",
        os.path.join(
            FIG_DIR,
            f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}"
            f"__CM_STRICT_WITH_UNKNOWN.png",
        ),
    )

    save_confusion(
        y_true,
        y_pred,
        ALLOWED_LABELS,
        f"{MODEL_NAME} Confusion Matrix (STRICT 5 labels)",
        os.path.join(
            FIG_DIR,
            f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}"
            f"__CM_STRICT_5LABELS.png",
        ),
    )

    covered_mask = [
        prediction != UNKNOWN_LABEL
        for prediction in y_pred
    ]
    covered_true = [
        gold
        for gold, keep in zip(y_true, covered_mask)
        if keep
    ]
    covered_pred = [
        prediction
        for prediction in y_pred
        if prediction != UNKNOWN_LABEL
    ]

    if covered_true:
        covered_accuracy = accuracy_score(
            covered_true,
            covered_pred,
        )
        covered_macro_f1 = f1_score(
            covered_true,
            covered_pred,
            labels=COVERED_LABELS,
            average="macro",
            zero_division=0,
        )
        covered_micro_f1 = f1_score(
            covered_true,
            covered_pred,
            labels=COVERED_LABELS,
            average="micro",
            zero_division=0,
        )
        covered_weighted_f1 = f1_score(
            covered_true,
            covered_pred,
            labels=COVERED_LABELS,
            average="weighted",
            zero_division=0,
        )
    else:
        covered_accuracy = 0.0
        covered_macro_f1 = 0.0
        covered_micro_f1 = 0.0
        covered_weighted_f1 = 0.0

    archive_path = get_prediction_archive_path(MODEL_NAME)
    save_partial_predictions(
        archive_path,
        MODEL_NAME,
        MODEL_CFG,
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
    )

    reason_counter = Counter(unknown_reasons)

    summary_rows.append({
        "provider": PROVIDER_NAME,
        "model_name": MODEL_NAME,
        "model_id": MODEL_CFG["model_id"],
        "run_id": RUN_ID,
        "setup": (
            "zero-shot + three-vote self-consistency (T=0.3) + "
            "model-specific Google output constraints + strict parser + "
            "retries + resume + autosave"
        ),
        "votes": VOTES,
        "temp_votes": FIXED_TEMPERATURE,
        "max_workers": MODEL_CFG["max_workers"],
        "initial_rps": MODEL_CFG["initial_rps"],
        "min_rps": MODEL_CFG["min_rps"],
        "max_rps": MODEL_CFG["max_rps"],
        "final_rps_estimate": rate_limiter.get_rps(),
        "structured_output": MODEL_CFG["use_structured_output"],
        "thinking_budget": MODEL_CFG["thinking_budget"],
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
        "unknown_reason_majority": int(
            reason_counter.get(FINAL_MAJORITY, 0)
        ),
        "unknown_reason_all_failed": int(
            reason_counter.get(FINAL_ALL_FAILED, 0)
        ),
        "unknown_reason_all_unparseable": int(
            reason_counter.get(FINAL_ALL_UNPARSEABLE, 0)
        ),
        "unknown_reason_all_different_111": int(
            reason_counter.get(FINAL_ALL_DIFFERENT_111, 0)
        ),
        "unknown_reason_no_majority_11u": int(
            reason_counter.get(FINAL_NO_MAJORITY_11U, 0)
        ),
    })


# =============================================================================
# SAVE SUMMARY
# =============================================================================
summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    METRIC_DIR,
    f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_THREE_VOTE.csv",
)
summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)

latest_summary_path = os.path.join(
    METRIC_DIR,
    f"{PROVIDER_NAME}__LATEST__SUMMARY_THREE_VOTE.csv",
)
summary_df.to_csv(
    latest_summary_path,
    index=False,
    encoding="utf-8-sig",
)

print(
    "\nGoogle three-vote benchmark completed successfully "
    "for all manuscript Google models."
)
print("RUN_ID:", RUN_ID)
print("Models run:", list(MODEL_CONFIGS.keys()))
print("Fixed temperature:", FIXED_TEMPERATURE)
print("Predictions:", PRED_DIR)
print("Figures    :", FIG_DIR)
print("Metrics    :", METRIC_DIR)
print("Summary    :", summary_path)