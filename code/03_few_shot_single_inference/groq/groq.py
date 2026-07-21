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

from groq import Groq


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

PROVIDER_NAME = "Groq"

MODELS = {
    "Qwen-3-32B": "qwen/qwen3-32b",
    "Allam-2-7B": "allam-2-7b",
    "LLaMA-3.1-8B": "llama-3.1-8b-instant",
}

# model-specific configs
MODEL_CONFIGS = {
    "Qwen-3-32B": {
        "max_workers": 2,
        "initial_rps": 0.8,
        "max_rps": 1.5,
        "request_spacing": 0.05,
        "use_system_msg": True,
        "max_tokens": 120,
        "reasoning_effort": "none",
    },
    "Allam-2-7B": {
        "max_workers": 2,
        "initial_rps": 1.2,
        "max_rps": 2.0,
        "request_spacing": 0.05,
        "use_system_msg": True,
        "max_tokens": 30,
        "reasoning_effort": None,
    },
    "LLaMA-3.1-8B": {
        "max_workers": 6,
        "initial_rps": 2.0,
        "max_rps": 4.0,
        "request_spacing": 0.0,
        "use_system_msg": True,
        "max_tokens": 30,
        "reasoning_effort": None,
    },
}

DEFAULT_MAX_WORKERS = 6
DEFAULT_INITIAL_RPS = 2.0
MIN_RPS = 0.2
DEFAULT_MAX_RPS = 4.0
DEFAULT_MAX_TOKENS = 30

MAX_RETRIES = 5
BASE_BACKOFF = 1.0
JITTER = 0.40

SEED = 42
random.seed(SEED)
DETERMINISTIC_JITTER = True

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__V2_FEWSHOT_SAFE"
PRINT_FIRST_N = 10
TEST_CALL_BEFORE_RUN = True

UNIFIED_PRED_COLS = [
    "text", "gold_label", "prediction",
    "abstained", "system_failed", "vote_pattern", "unknown_votes",
    "unknown_reason", "error_text", "calls_made",
    "provider", "model", "model_id", "run_id",
    "votes", "temp_votes", "max_workers",
]

# unknown reasons
UNK_API_ERROR = "api_error"
UNK_EMPTY_RESPONSE = "empty_response"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"
UNK_OK = "ok"


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
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Missing GROQ_API_KEY in environment (.env).")

client = Groq(api_key=api_key)


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
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "few_shot")
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
    raise RuntimeError(f"Gold labels contain values not in ALLOWED_LABELS: {bad_gold}\nAllowed: {ALLOWED_LABELS}")

df["emotion"].value_counts().to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__DATASET_class_distribution.csv"),
    encoding="utf-8-sig",
)


# ======================================================
# 3) Minimal Arabic Normalization
# ======================================================
def normalize_arabic(text):
    text = str(text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(normalize_arabic)


# ======================================================
# 4) FEW-SHOT EXAMPLES
# ======================================================
FEW_SHOTS = [
    {"text": normalize_arabic("اليوم فرحت كثير لما نجحت بالمقابله"), "label": "joy"},
    {"text": normalize_arabic("مستفز جدا وما عنده اي احترام"), "label": "anger"},
    {"text": normalize_arabic("حاسه بضيق وزعل من اللي صار"), "label": "sadness"},
    {"text": normalize_arabic("خايفه يصير اشي سيء بكرا"), "label": "fear"},
    {"text": normalize_arabic("رحت السوق واشتريت خبز وحليب"), "label": "neutral"},
]

SYSTEM_MSG = (
    "You are a strict emotion classifier for Arabic text. "
    "Return ONLY valid JSON. "
    "It MUST contain a key 'label'. "
    "The label MUST be one of: joy, anger, sadness, fear, neutral. "
    "Use lowercase only. "
    "If unsure, output neutral. "
    "Follow the labeled examples. "
    "No extra text."
)

def build_few_shot_prompt(text: str) -> str:
    parts = [
        "Choose exactly ONE label from: joy, anger, sadness, fear, neutral.",
        'Return ONLY JSON like: {"label":"joy"} (lowercase only).',
        'If unclear, return {"label":"neutral"}.',
        "",
        "Labeled examples:"
    ]

    for ex in FEW_SHOTS:
        parts.append(f'Sentence: "{ex["text"]}"')
        parts.append(f'Output: {{"label":"{ex["label"]}"}}')
        parts.append("")

    parts.append("Now classify this sentence:")
    parts.append(f'Sentence: "{text}"')

    return "\n".join(parts)


# ======================================================
# 5) ROBUST PARSER
# ======================================================
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
        repaired = blob.replace("'", '"')
        return json.loads(repaired)
    except Exception:
        return None


def _explicit_label_regex_fallback(raw: str) -> Optional[str]:
    if not raw:
        return None

    low = _strip_code_fences(raw).lower()

    m = re.search(
        r"""
        \blabel\b
        \s*[:=]\s*
        ["']?\s*
        (joy|anger|sadness|fear|neutral)
        \s*["']?
        """,
        low,
        flags=re.VERBOSE
    )
    if m:
        return m.group(1)

    return None


def parse_json_label_strict_with_reason(raw_text: str):
    raw = str(raw_text).strip() if raw_text is not None else ""

    if raw == "":
        return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT

    blob = _extract_first_json_object(raw)

    if blob is None:
        lab = _explicit_label_regex_fallback(raw)
        if lab is not None:
            return lab, "regex_label"
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


def _get_status_code(e: Exception) -> Optional[int]:
    sc = getattr(e, "status_code", None)
    if sc is not None:
        return sc
    resp = getattr(e, "response", None)
    if resp is not None:
        return getattr(resp, "status_code", None)
    return None


def _is_transient_status(status_code: Optional[int]) -> bool:
    return status_code in (408, 409, 429, 500, 502, 503, 504, 529) or status_code is None


# ======================================================
# 7) Call Groq
# ======================================================
def call_model(
    model_id,
    model_name,
    text,
    temperature,
    limiter: AdaptiveRateLimiter,
    request_spacing: float = 0.0,
    use_system_msg: bool = True,
    max_tokens: int = 30,
    reasoning_effort: Optional[str] = None,
    sample_i=None,
    vote_i=None
):
    prompt = build_few_shot_prompt(text)
    salt = f"{PROVIDER_NAME}|{model_name}|{sample_i}|{vote_i}"
    last_err = None
    attempts_used = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt
        try:
            limiter.wait()
            if request_spacing > 0:
                time.sleep(request_spacing)

            if use_system_msg:
                messages = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ]
            else:
                # fallback mode for models that may behave better without system
                messages = [
                    {"role": "user", "content": SYSTEM_MSG + "\n\n" + prompt},
                ]

            request_kwargs = {
                "model": model_id,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "messages": messages,
            }

            if reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = reasoning_effort

            resp = client.chat.completions.create(**request_kwargs)

            limiter.note_success()

            raw = ""
            if resp and getattr(resp, "choices", None):
                msg = resp.choices[0].message
                raw = getattr(msg, "content", "") or ""
            raw = str(raw).strip()

            if raw:
                return raw, 0, None, attempts_used

            return "", 1, "empty_message_text", attempts_used

        except Exception as e:
            status = _get_status_code(e)
            err_msg = str(e)[:300] if str(e) else e.__class__.__name__
            print(f"[API ERROR] model={model_name} attempt={attempt} status={status} err={err_msg}")

            last_err = f"status_{status}__{err_msg}" if status is not None else err_msg

            if status == 429:
                limiter.note_429(retry_after_seconds=None)

            if _is_transient_status(status):
                _sleep_backoff(attempt, salt)
                continue

            return "", 1, last_err, attempts_used

    return "", 1, (last_err or "max_retries_exceeded"), attempts_used


# ======================================================
# 8) Test call
# ======================================================
def run_test_call(model_id, model_name, cfg):
    print(f"\n[TEST] Testing {model_name} with 1 request...")

    limiter = AdaptiveRateLimiter(
        initial_rps=min(0.5, cfg["initial_rps"]),
        min_rps=MIN_RPS,
        max_rps=max(1.0, cfg["max_rps"])
    )

    test_text = normalize_arabic("اليوم فرحت كثير لما نجحت")
    raw, err_flag, err_text, calls_made = call_model(
        model_id=model_id,
        model_name=model_name,
        text=test_text,
        temperature=TEMP_VOTES,
        limiter=limiter,
        request_spacing=cfg["request_spacing"],
        use_system_msg=cfg["use_system_msg"],
        max_tokens=cfg["max_tokens"],
        reasoning_effort=cfg.get("reasoning_effort"),
        sample_i="TEST",
        vote_i=0
    )

    print(f"[TEST] err_flag: {err_flag} | attempts: {calls_made}")
    print(f"[TEST] err: {err_text}")
    print(f"[TEST] raw_head: {raw[:200] if raw else '<<EMPTY>>'}")

    if err_flag == 1:
        raise RuntimeError(f"Test call failed — API/HTTP error. Fix key/model/quota/prompt formatting.\n[TEST] err: {err_text}")

    pred, reason = parse_json_label_strict_with_reason(raw)
    print(f"[TEST] parsed: {pred} | reason: {reason}")

    if pred == UNKNOWN_LABEL:
        raise RuntimeError(f"Test call parsing failed for {model_name}. Raw head:\n{raw[:300]}")


# ======================================================
# 9) Vote=1 classification
# ======================================================
def classify_one(model_id, model_name, text, limiter: AdaptiveRateLimiter, cfg, sample_i=None):
    raw, err_flag, err_text, calls_made = call_model(
        model_id=model_id,
        model_name=model_name,
        text=text,
        temperature=TEMP_VOTES,
        limiter=limiter,
        request_spacing=cfg["request_spacing"],
        use_system_msg=cfg["use_system_msg"],
        max_tokens=cfg["max_tokens"],
        reasoning_effort=cfg.get("reasoning_effort"),
        sample_i=sample_i,
        vote_i=0
    )

    if err_flag == 1:
        pred = UNKNOWN_LABEL
        abstained = 1
        system_failed = 1
        unknown_votes = 1
        vote_pattern = "0"

        if err_text == "empty_message_text":
            reason = UNK_EMPTY_RESPONSE
        else:
            reason = UNK_API_ERROR

        return pred, 1, abstained, system_failed, calls_made, vote_pattern, unknown_votes, reason, str(err_text)[:500]

    pred, parse_reason = parse_json_label_strict_with_reason(raw)

    if pred == UNKNOWN_LABEL:
        abstained = 1
        system_failed = 0
        unknown_votes = 1
        vote_pattern = "0"
        reason = parse_reason
        return pred, 0, abstained, system_failed, calls_made, vote_pattern, unknown_votes, reason, str(raw)[:500]

    abstained = 0
    system_failed = 0
    unknown_votes = 0
    vote_pattern = "1"
    reason = UNK_OK
    return pred, 0, abstained, system_failed, calls_made, vote_pattern, unknown_votes, reason, None


# ======================================================
# 10) Plot confusion matrix
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
# 11) Run Benchmark per model
# ======================================================
summary_rows = []

for MODEL_NAME, MODEL_ID in MODELS.items():
    cfg = MODEL_CONFIGS.get(
        MODEL_NAME,
        {
            "max_workers": DEFAULT_MAX_WORKERS,
            "initial_rps": DEFAULT_INITIAL_RPS,
            "max_rps": DEFAULT_MAX_RPS,
            "request_spacing": 0.0,
            "use_system_msg": True,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "reasoning_effort": None,
        }
    )

    print(f"\nRunning {MODEL_NAME} (VOTES=1, TEMP=0, FEW-SHOT)...")
    print(
        f"Runtime config: workers={cfg['max_workers']}, "
        f"initial_rps={cfg['initial_rps']}, min_rps={MIN_RPS}, max_rps={cfg['max_rps']}, "
        f"request_spacing={cfg['request_spacing']}, use_system_msg={cfg['use_system_msg']}, "
        f"max_tokens={cfg['max_tokens']}, reasoning_effort={cfg.get('reasoning_effort')}"
    )

    if TEST_CALL_BEFORE_RUN:
        run_test_call(MODEL_ID, MODEL_NAME, cfg)

    LIMITER = AdaptiveRateLimiter(
        initial_rps=cfg["initial_rps"],
        min_rps=MIN_RPS,
        max_rps=cfg["max_rps"]
    )

    n = len(df)
    preds = [None] * n

    abstained_arr = [0] * n
    system_failed_arr = [0] * n
    vote_patterns = [None] * n
    unknown_votes_arr = [0] * n
    unknown_reasons = [None] * n
    error_texts = [None] * n
    calls_made_arr = [0] * n

    err_calls_total = 0
    abstention_total = 0
    system_fail_total = 0
    printed_count = 0

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = {
            executor.submit(classify_one, MODEL_ID, MODEL_NAME, txt, LIMITER, cfg, i): i
            for i, txt in enumerate(df["text"])
        }

        for fut in tqdm(as_completed(futures), total=n, desc=MODEL_NAME):
            i = futures[fut]

            pred, err_calls, abst_inc, sys_fail, calls_made, pat, unk_votes, reason, err_text = fut.result()

            preds[i] = pred
            abstained_arr[i] = abst_inc
            system_failed_arr[i] = sys_fail
            vote_patterns[i] = pat
            unknown_votes_arr[i] = int(unk_votes)
            unknown_reasons[i] = reason
            error_texts[i] = err_text
            calls_made_arr[i] = int(calls_made)

            err_calls_total += err_calls
            abstention_total += abst_inc
            system_fail_total += sys_fail

            if printed_count < PRINT_FIRST_N:
                print("-----")
                print(
                    f"[{i:04d}] GOLD={df['emotion'][i]}  "
                    f"PRED={pred}  abst={abst_inc}  sys_fail={sys_fail}  "
                    f"reason={reason}  calls={calls_made}"
                )
                print(f"TEXT: {df['text'][i]}")
                if err_text:
                    print(f"ERR: {err_text[:250]}")
                printed_count += 1

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
        title=f"{MODEL_NAME} Confusion Matrix (incl. unknown) — V2 Few-shot Safe",
        outpath=os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_WITH_UNKNOWN.png")
    )

    cm_5 = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
    save_confusion_matrix(
        cm_5,
        ALLOWED_LABELS,
        title=f"{MODEL_NAME} Confusion Matrix (5 labels) — V2 Few-shot Safe",
        outpath=os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_5LABELS.png")
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
        "text": df["text"],
        "gold_label": df["emotion"],
        "prediction": y_pred,

        "abstained": abstained_arr,
        "system_failed": system_failed_arr,
        "vote_pattern": vote_patterns,
        "unknown_votes": unknown_votes_arr,
        "unknown_reason": unknown_reasons,
        "error_text": error_texts,
        "calls_made": calls_made_arr,

        "provider": PROVIDER_NAME,
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "run_id": RUN_ID,

        "votes": VOTES,
        "temp_votes": TEMP_VOTES,
        "max_workers": cfg["max_workers"],
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
        "setup": "few-shot + single inference (V1) + model-specific runtime configuration",
        "votes": VOTES,
        "temp_votes": TEMP_VOTES,
        "max_workers": cfg["max_workers"],
        "total_samples": len(df),

        "strict_accuracy_unknown_wrong": strict_accuracy,
        "strict_macro_f1": strict_macro_f1,
        "strict_micro_f1": strict_micro_f1,
        "strict_weighted_f1": strict_weighted_f1,

        "coverage": coverage,
        "unknown_rate": unknown_rate,
        "unknown_total": int(unknown_total),

        "abstention_rate": abstention_total / len(df),
        "system_failure_rate": system_fail_total / len(df),
        "error_calls_per_sample": err_calls_total / len(df),

        "covered_accuracy": covered_accuracy,
        "covered_macro_f1": covered_macro_f1,
        "covered_micro_f1": covered_micro_f1,
        "covered_weighted_f1": covered_weighted_f1,

        "mean_calls_per_sample": sum(calls_made_arr) / len(calls_made_arr),
        "max_calls_for_any_sample": max(calls_made_arr),

        "unknown_reason_api_error": int(reason_counter.get(UNK_API_ERROR, 0)),
        "unknown_reason_empty_response": int(reason_counter.get(UNK_EMPTY_RESPONSE, 0)),
        "unknown_reason_invalid_json_no_object": int(reason_counter.get(UNK_INVALID_JSON_NO_OBJECT, 0)),
        "unknown_reason_invalid_json_parse_error": int(reason_counter.get(UNK_INVALID_JSON_PARSE_ERROR, 0)),
        "unknown_reason_invalid_json_missing_label": int(reason_counter.get(UNK_INVALID_JSON_MISSING_LABEL, 0)),
        "unknown_reason_invalid_label": int(reason_counter.get(UNK_INVALID_LABEL, 0)),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_V2_fewshot_safe.csv"),
    index=False,
    encoding="utf-8-sig",
)

print("\nGroq few-shot benchmark completed successfully for all manuscript models (VOTE=1, TEMP=0).")
print("RUN_ID:", RUN_ID)
print("Models run:", list(MODELS.keys()))
print("Outputs saved under:")
print(" - predictions:", PRED_DIR)
print(" - figures    :", FIG_DIR)
print(" - metrics    :", METRIC_DIR)