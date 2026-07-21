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

# avoid tkinter/thread issues on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import anthropic


# ===============================
# CONFIG (VOTE=1, STRICT JSON UNKNOWN) — UNIFIED STANDARD
# ===============================
VOTES = 1

# best for classification stability (esp. vote=1)
TEMP_VOTES = 0.0  # temperature 

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_LABELS = ALLOWED_LABELS + [UNKNOWN_LABEL]
COVERED_LABELS = ALLOWED_LABELS
STRICT_UNKNOWN_SENTINEL = "__unknown__"

PROVIDER_NAME = "Claude"

MODELS = {
    "Claude-Haiku-3": "claude-3-haiku-20240307",
    "Claude-Sonnet-4.6": "claude-sonnet-4-6",
    # add if you have access:
    # "Claude-Opus-4.6": "claude-opus-4-6",
}

# FAST + STABLE defaults (tweak if you see many 429)
MAX_WORKERS = 8
MAX_RETRIES = 5
BASE_BACKOFF = 1.0
JITTER = 0.40
REQUEST_SPACING = 0.0

# limiter knobs
INITIAL_RPS = 2.5
MIN_RPS = 0.2
MAX_RPS = 6.0

SEED = 42
random.seed(SEED)
DETERMINISTIC_JITTER = True

# Vote list is not very useful for VOTES=1 (kept optional)
STORE_VOTES_LIST = False

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__V1"


# ===============================
# UNIFIED OUTPUT SCHEMA
# ===============================
UNIFIED_PRED_COLS = [
    "text", "gold_label", "prediction",
    "abstained", "vote_pattern", "unknown_votes", "unknown_reason", "calls_made",
    "provider", "model", "run_id",
    "votes", "temp_votes", "max_workers",
]


# ===============================
# UNIFIED unknown reasons
# ===============================
UNK_API_ERROR = "api_error"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"
UNK_OK = "ok"


# ======================================================
# FLOAT-AWARE GLOBAL RATE LIMITER (ANTI-429)
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
                extra = min(10.0, 1.5 * self._consecutive_429)

            self._cooldown_until = max(self._cooldown_until, time.time() + extra)

    def note_success(self):
        with self._lock:
            if self._consecutive_429 > 0:
                self._consecutive_429 = max(0, self._consecutive_429 - 1)
            self._rps = min(self._max_rps, self._rps + 0.05)


# ======================================================
# 1) Load API
# ======================================================
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("Missing ANTHROPIC_API_KEY in environment (.env).")

client = anthropic.Anthropic(api_key=api_key)


# ======================================================
# 2) Paths
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
FIG_DIR = os.path.join(OUTPUT_ROOT, "figures")
METRIC_DIR = os.path.join(OUTPUT_ROOT, "metrics")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8")


# ======================================================
# 2.1) Data hygiene
# ======================================================
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

df["emotion"].value_counts().to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__DATASET_class_distribution.csv"),
    encoding="utf-8-sig",
)


# ======================================================
# 3) Minimal Arabic Normalization (keep light)
# ======================================================
def normalize_arabic(text):
    text = str(text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(normalize_arabic)


# ======================================================
# 4) Strict JSON Prompt (UNIFIED)
# IMPORTANT: if unsure -> neutral (NOT unknown)
# ======================================================
SYSTEM_MSG = (
    "You are a strict emotion classifier for Arabic text. "
    "Return ONLY valid JSON. "
    "It MUST contain a key 'label'. "
    "The label MUST be one of: joy, anger, sadness, fear, neutral. "
    "Use lowercase only. "
    "If unsure, output neutral. "
    "No extra text."
)

PROMPT = """
Choose exactly ONE label from: joy, anger, sadness, fear, neutral.
Return ONLY JSON like: {"label":"joy"} (lowercase only).
If unclear, return {"label":"neutral"}.

Sentence: "{TEXT}"
""".strip()


# ======================================================
# 5) ROBUST PARSER (RESEARCH-GRADE + FAIR)
# Fixed fallback: only triggers on explicit label patterns
# ======================================================

def clean_label(lbl):
    """
    Normalize label safely:
    - lower
    - remove punctuation
    - accept exact token or label embedded as a word
    """
    if lbl is None:
        return UNKNOWN_LABEL

    s = str(lbl).strip().lower()

    # remove punctuation except underscore (safe)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return UNKNOWN_LABEL

    # first token
    first = s.split(" ")[0]
    if first in ALLOWED_LABELS:
        return first

    # search embedded label as a whole word
    for lab in ALLOWED_LABELS:
        if re.search(rf"\b{lab}\b", s):
            return lab

    return UNKNOWN_LABEL


def _strip_code_fences(text: str) -> str:
    # remove markdown fences: ```json ... ``` or ``` ... ```
    t = "" if text is None else str(text)
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```", "", t)
    return t.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract first {...} block robustly (non-greedy).
    """
    if not text:
        return None

    t = _strip_code_fences(text)
    m = re.search(r"\{.*?\}", t, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0).strip()


def _try_load_json_obj(blob: str):
    """
    Try JSON loads; then minimal repair for single quotes.
    """
    try:
        return json.loads(blob)
    except Exception:
        pass

    # minimal repair: {'label':'joy'} -> {"label":"joy"}
    try:
        repaired = blob.replace("'", '"')
        return json.loads(repaired)
    except Exception:
        return None


def _explicit_label_regex_fallback(raw: str) -> Optional[str]:
    """
    FAIR fallback: only accept explicit label patterns.
    Examples accepted:
      label: joy
      label = "joy"
      "label":"joy"
      'label' : 'joy'
    NOT accepted:
      "choose one of joy, anger, sadness..."  (no explicit label assignment)
    """
    if not raw:
        return None

    low = _strip_code_fences(raw).lower()

    # strict explicit patterns
    m = re.search(
        r"""
        \blabel\b              # key
        \s*[:=]\s*             # separator
        ["']?\s*               # optional quote + spaces
        (joy|anger|sadness|fear|neutral)  # label
        \s*["']?               # optional quote
        """,
        low,
        flags=re.VERBOSE
    )

    if m:
        return m.group(1)

    return None


def parse_json_label_strict_with_reason(raw_text: str):
    """
    Returns: (label, reason)
    Reasons:
      ok
      regex_label
      invalid_json_no_object
      invalid_json_parse_error
      invalid_json_missing_label
      invalid_label
    """
    raw = str(raw_text).strip() if raw_text is not None else ""

    if raw == "":
        return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT

    # 1) Extract JSON object
    blob = _extract_first_json_object(raw)

    # 2) If no JSON object, use FAIR explicit regex fallback
    if blob is None:
        lab = _explicit_label_regex_fallback(raw)
        if lab is not None:
            return lab, "regex_label"
        return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT

    # 3) Load JSON (with minimal repair)
    obj = _try_load_json_obj(blob)
    if obj is None:
        return UNKNOWN_LABEL, UNK_INVALID_JSON_PARSE_ERROR

    if not isinstance(obj, dict):
        return UNKNOWN_LABEL, UNK_INVALID_JSON_MISSING_LABEL

    # 4) Find label key case-insensitive
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
# 6) Backoff helpers
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

def _is_transient_status(status_code: Optional[int]) -> bool:
    # Claude: 429 rate_limit_error, 500 api_error, 529 overloaded_error
    return status_code in (408, 429, 500, 502, 503, 504, 529)


# ======================================================
# 7) Call Claude
# ======================================================
def _extract_text_from_message(message_obj) -> str:
    chunks = []
    content = getattr(message_obj, "content", None)

    if isinstance(content, list):
        for b in content:
            if hasattr(b, "text"):
                chunks.append(b.text)
            elif isinstance(b, dict) and "text" in b:
                chunks.append(str(b["text"]))
    elif content is not None:
        chunks.append(str(content))

    return "\n".join([c for c in chunks if c is not None]).strip()

def call_model(model_id, model_name, text, temperature, limiter: AdaptiveRateLimiter, sample_i=None, vote_i=None):
    prompt = PROMPT.replace("{TEXT}", text)
    salt = f"{PROVIDER_NAME}|{model_name}|{sample_i}|{vote_i}"
    last_err = None
    attempts_used = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt
        try:
            limiter.wait()
            if REQUEST_SPACING > 0:
                time.sleep(REQUEST_SPACING)

            msg = client.messages.create(
                model=model_id,
                max_tokens=20,
                temperature=float(temperature),
                system=SYSTEM_MSG,
                messages=[{"role": "user", "content": prompt}],
            )

            limiter.note_success()
            raw = _extract_text_from_message(msg)
            if raw:
                return raw, 0, None, attempts_used

            return "", 1, "empty_message_text", attempts_used

        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            last_err = f"status_{status}"

            retry_after_s = None
            try:
                headers = getattr(getattr(e, "response", None), "headers", None)
                if headers:
                    ra = headers.get("retry-after") or headers.get("Retry-After")
                    if ra is not None:
                        retry_after_s = float(ra)
            except Exception:
                retry_after_s = None

            if status == 429:
                limiter.note_429(retry_after_seconds=retry_after_s)

            if _is_transient_status(status):
                if retry_after_s is not None:
                    j = _det_jitter(SEED, attempt, salt) if DETERMINISTIC_JITTER else random.uniform(0, JITTER)
                    time.sleep(max(0.0, retry_after_s) + j)
                else:
                    _sleep_backoff(attempt, salt)
                continue

            return "", 1, last_err, attempts_used

        except anthropic.APIConnectionError as e:
            last_err = ("conn_" + str(e))[:200]
            _sleep_backoff(attempt, salt)
            continue

        except Exception as e:
            last_err = str(e)[:200]
            _sleep_backoff(attempt, salt)
            continue

    return "", 1, (last_err or "max_retries_exceeded"), attempts_used


# ======================================================
# 8) Vote=1 classification
# ======================================================
def classify_one(model_id, model_name, text, limiter: AdaptiveRateLimiter, sample_i=None):
    raw, err_flag, err_text, calls_made = call_model(
        model_id, model_name, text,
        temperature=TEMP_VOTES,
        limiter=limiter,
        sample_i=sample_i,
        vote_i=0
    )

    if err_flag == 1:
        pred = UNKNOWN_LABEL
        abstained = 1
        unknown_votes = 1
        vote_pattern = "0"
        reason = UNK_API_ERROR
        votes_list = [f"__API_ERROR__ {str(err_text)[:200]}"]
        return pred, 1, abstained, calls_made, votes_list, vote_pattern, unknown_votes, reason

    pred, parse_reason = parse_json_label_strict_with_reason(raw)
    votes_list = [pred]

    if pred == UNKNOWN_LABEL:
        abstained = 1
        unknown_votes = 1
        vote_pattern = "0"
        reason = parse_reason
        return pred, 0, abstained, calls_made, votes_list, vote_pattern, unknown_votes, reason

    abstained = 0
    unknown_votes = 0
    vote_pattern = "1"
    reason = UNK_OK
    return pred, 0, abstained, calls_made, votes_list, vote_pattern, unknown_votes, reason


# ======================================================
# 9) Plot confusion matrix (matplotlib only)
# ======================================================
def save_confusion_matrix(cm, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # numbers
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
# 10) Run Benchmark per model
# ======================================================
summary_rows = []

for MODEL_NAME, MODEL_ID in MODELS.items():
    print(f"\nRunning {MODEL_NAME} (VOTES=1, TEMP=0)...")

    LIMITER = AdaptiveRateLimiter(initial_rps=INITIAL_RPS, min_rps=MIN_RPS, max_rps=MAX_RPS)

    n = len(df)
    preds = [None] * n

    abstained_arr = [0] * n
    vote_patterns = [None] * n
    unknown_votes_arr = [0] * n
    unknown_reasons = [None] * n
    calls_made_arr = [0] * n
    votes_list_arr = [None] * n

    err_calls_total = 0
    abstention_total = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_one, MODEL_ID, MODEL_NAME, txt, LIMITER, i): i
            for i, txt in enumerate(df["text"])
        }

        for fut in tqdm(as_completed(futures), total=n, desc=MODEL_NAME):
            i = futures[fut]
            pred, err_calls, abst_inc, calls_made, votes_list, pat, unk_votes, reason = fut.result()

            preds[i] = pred
            abstained_arr[i] = abst_inc
            vote_patterns[i] = pat
            unknown_votes_arr[i] = int(unk_votes)
            unknown_reasons[i] = reason
            calls_made_arr[i] = int(calls_made)

            if STORE_VOTES_LIST:
                votes_list_arr[i] = json.dumps(votes_list, ensure_ascii=False)

            err_calls_total += err_calls
            abstention_total += abst_inc

    y_true = df["emotion"].tolist()
    y_pred = preds

    unknown_total = y_pred.count(UNKNOWN_LABEL)
    unknown_rate = unknown_total / len(y_pred)
    coverage = 1 - unknown_rate

    # unknown reason distribution (important for paper)
    pd.Series(unknown_reasons).value_counts(dropna=False).to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__UNKNOWN_REASON_DISTRIBUTION.csv"),
        encoding="utf-8-sig",
    )

    # STRICT (unknown=wrong) for metrics
    y_pred_strict = [p if p != UNKNOWN_LABEL else STRICT_UNKNOWN_SENTINEL for p in y_pred]

    strict_accuracy = accuracy_score(y_true, y_pred_strict)
    strict_macro_f1 = f1_score(y_true, y_pred_strict, average="macro", labels=ALLOWED_LABELS, zero_division=0)
    strict_micro_f1 = f1_score(y_true, y_pred_strict, average="micro", labels=ALLOWED_LABELS, zero_division=0)
    strict_weighted_f1 = f1_score(y_true, y_pred_strict, average="weighted", labels=ALLOWED_LABELS, zero_division=0)

    # Per-class report (5 labels) — GOOD for paper
    report_strict_5 = classification_report(
        y_true, y_pred_strict, labels=ALLOWED_LABELS, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_strict_5).transpose().to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__REPORT_STRICT_5LABELS.csv"),
        encoding="utf-8-sig",
    )

    # Confusion matrices (paper figures)
    cm_strict_with_unknown = confusion_matrix(y_true, y_pred, labels=STRICT_LABELS)
    save_confusion_matrix(
        cm_strict_with_unknown,
        STRICT_LABELS,
        title=f"{MODEL_NAME} Confusion Matrix (incl. unknown) — V1",
        outpath=os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_WITH_UNKNOWN.png")
    )

    cm_5 = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
    save_confusion_matrix(
        cm_5,
        ALLOWED_LABELS,
        title=f"{MODEL_NAME} Confusion Matrix (5 labels) — V1",
        outpath=os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_5LABELS.png")
    )

    # COVERED metrics (exclude unknown)
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

    # Save predictions (UNIFIED)
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
        "run_id": RUN_ID,

        "votes": VOTES,
        "temp_votes": TEMP_VOTES,
        "max_workers": MAX_WORKERS,
    })

    if STORE_VOTES_LIST:
        out_pred_df["votes_list"] = votes_list_arr
        cols = UNIFIED_PRED_COLS + ["votes_list"]
    else:
        cols = UNIFIED_PRED_COLS

    out_pred_df = out_pred_df[cols]
    out_pred_df.to_csv(
        os.path.join(PRED_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__PREDICTIONS.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    reason_counter = Counter(unknown_reasons)

    summary_rows.append({
        "provider": PROVIDER_NAME,
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "setup": "zero-shot + single inference (V1)",
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
        "abstention_rate": abstention_total / len(df),
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
    })

# Provider-level summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_V1_strict_json.csv"),
    index=False,
    encoding="utf-8-sig",
)

print("\nClaude benchmark completed successfully (VOTE=1, TEMP=0, STRICT JSON unknown).")
print("RUN_ID:", RUN_ID)
print("Models run:", list(MODELS.keys()))
print("Outputs saved under:")
print(" - predictions:", PRED_DIR)
print(" - figures    :", FIG_DIR)
print(" - metrics    :", METRIC_DIR)