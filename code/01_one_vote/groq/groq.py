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
from typing import Optional, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from groq import Groq

# avoid tkinter/thread issues on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================
# 0) Global Config
# ======================================================
SEED = 42
random.seed(SEED)

PROVIDER_NAME = "Groq"

VOTES = 1
TEMP_VOTES = 0.0
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__V1"

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_LABELS = ALLOWED_LABELS + [UNKNOWN_LABEL]

# retry / backoff
MAX_RETRIES = 4
BASE_BACKOFF = 1.4
JITTER = 0.6
DETERMINISTIC_JITTER = True

# plotting / output
SAVE_CONFUSION_MATRIX = True
SAVE_CLASSIFICATION_REPORT = True

# unknown reasons
UNK_OK = "ok"
UNK_REGEX_LABEL = "regex_label"
UNK_API_ERROR = "api_error"
UNK_INVALID_JSON_NO_OBJECT = "invalid_json_no_object"
UNK_INVALID_JSON_PARSE_ERROR = "invalid_json_parse_error"
UNK_INVALID_JSON_MISSING_LABEL = "invalid_json_missing_label"
UNK_INVALID_LABEL = "invalid_label"

# debug: print first few raw outputs per model
DEBUG_PRINT_FIRST_N_RAW = 5


# ======================================================
# Per-model config
# ======================================================
MODEL_CONFIGS = {
    "Qwen-3-32B": {
        "model_id": "qwen/qwen3-32b",
        "max_workers": 2,
        "initial_rps": 0.8,
        "min_rps": 0.2,
        "max_rps": 1.5,
        "request_spacing": 0.05,
        "max_tokens": 120,
        "reasoning_effort": "none",
        "system_msg": (
            "You are a strict classification system for Arabic emotion classification. "
            "Return exactly one JSON object and nothing else. "
            "Do not explain. "
            "Do not output markdown. "
            "The JSON must contain exactly one key: label. "
            "The label must be one of: joy, anger, sadness, fear, neutral. "
            "Use lowercase only. "
            "If unsure, return {\"label\":\"neutral\"}."
        ),
        "prompt": """
Classify the emotion of the following Arabic sentence.

Allowed labels:
joy, anger, sadness, fear, neutral

Rules:
- Output valid JSON only
- Output exactly one object
- Output exactly one key: "label"
- No explanation
- No extra text

Required format:
{"label":"joy"}

Sentence: "{TEXT}"
""".strip(),
    },
    "Allam-2-7B": {
        "model_id": "allam-2-7b",
        "max_workers": 2,
        "initial_rps": 0.8,
        "min_rps": 0.2,
        "max_rps": 1.5,
        "request_spacing": 0.05,
        "max_tokens": 120,
        "reasoning_effort": None,
        "system_msg": (
            "You are a strict classification system for Arabic emotion classification. "
            "Return exactly one JSON object and nothing else. "
            "Do not explain. "
            "Do not output markdown. "
            "The JSON must contain exactly one key: label. "
            "The label must be one of: joy, anger, sadness, fear, neutral. "
            "Use lowercase only. "
            "If unsure, return {\"label\":\"neutral\"}."
        ),
        "prompt": """
Classify the emotion of the following Arabic sentence.

Allowed labels:
joy, anger, sadness, fear, neutral

Rules:
- Output valid JSON only
- Output exactly one object
- Output exactly one key: "label"
- No explanation
- No extra text

Required format:
{"label":"joy"}

Sentence: "{TEXT}"
""".strip(),
    },
    "LLaMA-3.1-8B": {
        "model_id": "llama-3.1-8b-instant",
        "max_workers": 2,
        "initial_rps": 0.8,
        "min_rps": 0.2,
        "max_rps": 1.5,
        "request_spacing": 0.05,
        "max_tokens": 120,
        "reasoning_effort": None,
        "system_msg": (
            "You are a strict classification system for Arabic emotion classification. "
            "Return exactly one JSON object and nothing else. "
            "Do not explain. "
            "Do not output markdown. "
            "The JSON must contain exactly one key: label. "
            "The label must be one of: joy, anger, sadness, fear, neutral. "
            "Use lowercase only. "
            "If unsure, return {\"label\":\"neutral\"}."
        ),
        "prompt": """
Classify the emotion of the following Arabic sentence.

Allowed labels:
joy, anger, sadness, fear, neutral

Rules:
- Output valid JSON only
- Output exactly one object
- Output exactly one key: "label"
- No explanation
- No extra text

Required format:
{"label":"joy"}

Sentence: "{TEXT}"
""".strip(),
    },
}


# ======================================================
# Adaptive Rate Limiter
# ======================================================
class AdaptiveRateLimiter:
    def __init__(self, initial_rps=2.0, min_rps=0.2, max_rps=10.0):
        self._rps = float(initial_rps)
        self._min_rps = float(min_rps)
        self._max_rps = float(max_rps)
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._consecutive_429 = 0

    def wait(self):
        with self._lock:
            now = time.time()
            wait_s = max(0.0, self._next_allowed - now)
        if wait_s > 0:
            time.sleep(wait_s)

        with self._lock:
            now = time.time()
            interval = 1.0 / max(self._rps, 1e-6)
            self._next_allowed = max(self._next_allowed, now) + interval

    def note_429(self):
        with self._lock:
            self._consecutive_429 += 1
            factor = 0.75 if self._consecutive_429 < 3 else 0.60
            self._rps = max(self._min_rps, self._rps * factor)
            extra = min(2.0, 0.2 * self._consecutive_429)
            self._next_allowed = max(self._next_allowed, time.time() + extra)

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
# 4) Robust Parser
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


def _strip_think_blocks(text: str) -> str:
    t = "" if text is None else str(text)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)
    return t.strip()


def _strip_code_fences(text: str) -> str:
    t = "" if text is None else str(text)
    t = _strip_think_blocks(t)
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```", "", t)
    return t.strip()


def _extract_last_json_object(text: str) -> Optional[str]:
    """
    Extract last {...} block robustly.
    This helps when the model emits extra content before the final JSON.
    """
    if not text:
        return None

    t = _strip_code_fences(text)
    matches = re.findall(r"\{.*?\}", t, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


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
    """
    FAIR fallback: only accept explicit label assignment patterns.
    """
    if not raw:
        return None

    low = _strip_code_fences(raw).lower()

    # explicit label assignment
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

    # optional fallback for bare label only
    # keep this enabled for Qwen rescue; disable if you want stricter paper runs
    m2 = re.fullmatch(r"\s*(joy|anger|sadness|fear|neutral)\s*", low)
    if m2:
        return m2.group(1)

    return None


def parse_json_label_strict_with_reason(raw_text: str):
    """
    Returns: (label, reason)
    """
    raw = str(raw_text).strip() if raw_text is not None else ""

    if raw == "":
        return UNKNOWN_LABEL, UNK_INVALID_JSON_NO_OBJECT

    fallback_label = _explicit_label_regex_fallback(raw)
    if fallback_label is not None:
        return fallback_label, UNK_REGEX_LABEL

    blob = _extract_last_json_object(raw)
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

    lab = clean_label(obj.get(label_key))
    if lab == UNKNOWN_LABEL:
        return UNKNOWN_LABEL, UNK_INVALID_LABEL

    return lab, UNK_OK


# ======================================================
# 5) Backoff helpers
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
# 6) Call Groq
# ======================================================
def call_model(
    model_cfg: Dict[str, Any],
    model_name: str,
    text: str,
    temperature: float,
    limiter: AdaptiveRateLimiter,
    sample_i=None,
    vote_i=None
):
    model_id = model_cfg["model_id"]
    prompt = model_cfg["prompt"].replace("{TEXT}", text)
    system_msg = model_cfg["system_msg"]
    max_tokens = int(model_cfg["max_tokens"])
    request_spacing = float(model_cfg["request_spacing"])

    salt = f"{PROVIDER_NAME}|{model_name}|{sample_i}|{vote_i}"
    attempts_used = 0
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt
        try:
            limiter.wait()
            if request_spacing > 0:
                time.sleep(request_spacing)

            request_kwargs = {
                "model": model_id,
                "temperature": float(temperature),
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
            }

            reasoning_effort = model_cfg.get("reasoning_effort")
            if reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = reasoning_effort

            resp = client.chat.completions.create(**request_kwargs)

            limiter.note_success()

            raw = ""
            if resp and getattr(resp, "choices", None):
                msg = resp.choices[0].message
                raw = getattr(msg, "content", "") or ""
            raw = str(raw).strip()

            if sample_i is not None and sample_i < DEBUG_PRINT_FIRST_N_RAW:
                head = raw[:500].replace("\n", "\\n")
                print(f"[RAW][{model_name}][{sample_i}] {head}")

            if raw:
                return raw, 0, None, attempts_used

            return "", 1, "empty_message_text", attempts_used

        except Exception as e:
            last_err = str(e)
            status_code = _get_status_code(e)

            if status_code == 429:
                limiter.note_429()

            if attempt < MAX_RETRIES and _is_transient_status(status_code):
                _sleep_backoff(attempt, salt)
                continue

            return "", 1, f"api_error: {last_err}", attempts_used

    return "", 1, f"api_error: {last_err}", attempts_used


# ======================================================
# 7) One sample
# ======================================================
def run_single_sample(i, row, model_cfg, model_name, limiter):
    text = row["text"]
    gold = row["emotion"]

    raw, err_flag, err_text, calls_made = call_model(
        model_cfg=model_cfg,
        model_name=model_name,
        text=text,
        temperature=TEMP_VOTES,
        limiter=limiter,
        sample_i=i,
        vote_i=1,
    )

    if err_flag == 1:
        pred = UNKNOWN_LABEL
        reason = UNK_API_ERROR
    else:
        pred, reason = parse_json_label_strict_with_reason(raw)

    return {
        "index": i,
        "text": text,
        "gold": gold,
        "pred": pred,
        "abstained": int(pred == UNKNOWN_LABEL),
        "reason": reason,
        "votes": 1,
        "vote_1": pred,
        "vote_1_raw": raw,
        "calls": calls_made,
        "err_flag": err_flag,
        "err_text": err_text,
        "correct": int(pred == gold),
        "provider": PROVIDER_NAME,
        "model": model_name,
        "run_id": RUN_ID,
    }


# ======================================================
# 8) Metrics helpers
# ======================================================
def safe_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=ALLOWED_LABELS, average="macro", zero_division=0)


def safe_micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=ALLOWED_LABELS, average="micro", zero_division=0)


def safe_weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=ALLOWED_LABELS, average="weighted", zero_division=0)


def compute_metrics(pred_df: pd.DataFrame):
    y_true = pred_df["gold"].tolist()
    y_pred = pred_df["pred"].tolist()

    strict_accuracy = accuracy_score(y_true, y_pred)
    strict_macro_f1 = safe_macro_f1(y_true, y_pred)
    strict_micro_f1 = safe_micro_f1(y_true, y_pred)
    strict_weighted_f1 = safe_weighted_f1(y_true, y_pred)

    covered_df = pred_df[pred_df["pred"] != UNKNOWN_LABEL].copy()
    if len(covered_df) > 0:
        covered_accuracy = accuracy_score(covered_df["gold"], covered_df["pred"])
        covered_macro_f1 = safe_macro_f1(covered_df["gold"], covered_df["pred"])
        covered_micro_f1 = safe_micro_f1(covered_df["gold"], covered_df["pred"])
        covered_weighted_f1 = safe_weighted_f1(covered_df["gold"], covered_df["pred"])
    else:
        covered_accuracy = 0.0
        covered_macro_f1 = 0.0
        covered_micro_f1 = 0.0
        covered_weighted_f1 = 0.0

    unknown_total = int((pred_df["pred"] == UNKNOWN_LABEL).sum())
    abstention_total = unknown_total
    coverage = 1.0 - (unknown_total / len(pred_df))
    unknown_rate = unknown_total / len(pred_df)

    return {
        "strict_accuracy": strict_accuracy,
        "strict_macro_f1": strict_macro_f1,
        "strict_micro_f1": strict_micro_f1,
        "strict_weighted_f1": strict_weighted_f1,
        "coverage": coverage,
        "unknown_rate": unknown_rate,
        "unknown_total": unknown_total,
        "abstention_total": abstention_total,
        "covered_accuracy": covered_accuracy,
        "covered_macro_f1": covered_macro_f1,
        "covered_micro_f1": covered_micro_f1,
        "covered_weighted_f1": covered_weighted_f1,
    }


def save_confusion(y_true, y_pred, out_png, title):
    labels = ALLOWED_LABELS + [UNKNOWN_LABEL]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ======================================================
# 9) Main run per model
# ======================================================
summary_rows = []

for model_name, model_cfg in MODEL_CONFIGS.items():
    print("\n" + "=" * 80)
    print(f"Running {model_name} (VOTES={VOTES}, TEMP={TEMP_VOTES})...")

    max_workers = int(model_cfg["max_workers"])
    initial_rps = float(model_cfg["initial_rps"])
    min_rps = float(model_cfg["min_rps"])
    max_rps = float(model_cfg["max_rps"])
    request_spacing = float(model_cfg["request_spacing"])
    max_tokens = int(model_cfg["max_tokens"])

    print(
        f"Runtime config: workers={max_workers}, initial_rps={initial_rps}, "
        f"min_rps={min_rps}, max_rps={max_rps}, temperature={TEMP_VOTES}, "
        f"request_spacing={request_spacing}, max_output_tokens={max_tokens}"
    )

    limiter = AdaptiveRateLimiter(
        initial_rps=initial_rps,
        min_rps=min_rps,
        max_rps=max_rps
    )

    records = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, row in df.iterrows():
            futures.append(ex.submit(run_single_sample, i, row, model_cfg, model_name, limiter))

        for fut in tqdm(as_completed(futures), total=len(futures), desc=model_name):
            records.append(fut.result())

    pred_df = pd.DataFrame(records).sort_values("index").reset_index(drop=True)

    pred_path = os.path.join(
        PRED_DIR,
        f"{PROVIDER_NAME}__{model_name}__RUN{RUN_ID}__predictions_V1.csv"
    )
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    metrics = compute_metrics(pred_df)

    if SAVE_CLASSIFICATION_REPORT:
        report = classification_report(
            pred_df["gold"],
            pred_df["pred"],
            labels=ALLOWED_LABELS + [UNKNOWN_LABEL],
            zero_division=0,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(
            METRIC_DIR,
            f"{PROVIDER_NAME}__{model_name}__RUN{RUN_ID}__classification_report_V1.csv"
        )
        report_df.to_csv(report_path, encoding="utf-8-sig")

    if SAVE_CONFUSION_MATRIX:
        fig_path = os.path.join(
            FIG_DIR,
            f"{PROVIDER_NAME}__{model_name}__RUN{RUN_ID}__confusion_matrix_V1.png"
        )
        save_confusion(
            y_true=pred_df["gold"].tolist(),
            y_pred=pred_df["pred"].tolist(),
            out_png=fig_path,
            title=f"{model_name} confusion matrix"
        )

    reason_counter = Counter(pred_df["reason"].tolist())
    err_calls_total = int(pred_df["err_flag"].sum())

    summary_rows.append({
        "provider": PROVIDER_NAME,
        "model_name": model_name,
        "run_id": RUN_ID,
        "setup": "zero-shot + single inference (V1)",
        "votes": VOTES,
        "temp_votes": TEMP_VOTES,
        "max_workers": max_workers,
        "total_samples": len(df),

        "strict_accuracy_unknown_wrong": metrics["strict_accuracy"],
        "strict_macro_f1": metrics["strict_macro_f1"],
        "strict_micro_f1": metrics["strict_micro_f1"],
        "strict_weighted_f1": metrics["strict_weighted_f1"],

        "coverage": metrics["coverage"],
        "unknown_rate": metrics["unknown_rate"],
        "unknown_total": int(metrics["unknown_total"]),
        "abstention_rate": metrics["abstention_total"] / len(df),
        "error_calls_per_sample": err_calls_total / len(df),

        "covered_accuracy": metrics["covered_accuracy"],
        "covered_macro_f1": metrics["covered_macro_f1"],
        "covered_micro_f1": metrics["covered_micro_f1"],
        "covered_weighted_f1": metrics["covered_weighted_f1"],

        "unknown_reason_api_error": int(reason_counter.get(UNK_API_ERROR, 0)),
        "unknown_reason_invalid_json_no_object": int(reason_counter.get(UNK_INVALID_JSON_NO_OBJECT, 0)),
        "unknown_reason_invalid_json_parse_error": int(reason_counter.get(UNK_INVALID_JSON_PARSE_ERROR, 0)),
        "unknown_reason_invalid_json_missing_label": int(reason_counter.get(UNK_INVALID_JSON_MISSING_LABEL, 0)),
        "unknown_reason_invalid_label": int(reason_counter.get(UNK_INVALID_LABEL, 0)),
        "unknown_reason_regex_label": int(reason_counter.get(UNK_REGEX_LABEL, 0)),
        "predictions_path": pred_path,
    })

    print(f"\n{model_name} summary:")
    print(f"  Strict Acc      : {metrics['strict_accuracy']:.4f}")
    print(f"  Strict Macro-F1 : {metrics['strict_macro_f1']:.4f}")
    print(f"  Coverage        : {metrics['coverage']:.4f}")
    print(f"  Unknown Rate    : {metrics['unknown_rate']:.4f}")
    print(f"  Unknown Total   : {metrics['unknown_total']}")


summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(
    METRIC_DIR,
    f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_V1_strict_json.csv"
)
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\nGroq benchmark completed successfully for all manuscript Groq models (VOTE=1, TEMP=0).")
print("RUN_ID:", RUN_ID)
print("Models run:", list(MODEL_CONFIGS.keys()))
print("Outputs saved under:")
print(" - predictions:", PRED_DIR)
print(" - figures    :", FIG_DIR)
print(" - metrics    :", METRIC_DIR)
print(" - summary    :", summary_path)