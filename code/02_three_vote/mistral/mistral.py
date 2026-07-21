import os
import re
import time
import json
import random
import hashlib
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from mistralai import Mistral
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================
# UNIFIED CONFIG — 3 VOTES, NO TIEBREAK
# ======================================================
VOTES = 3
TEMP_VOTES = 0.3

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_LABELS = ALLOWED_LABELS + [UNKNOWN_LABEL]
COVERED_LABELS = ALLOWED_LABELS
STRICT_UNKNOWN_SENTINEL = "__unknown__"

PROVIDER_NAME = "Mistral"

MODELS = {
    "Mistral-Medium": "mistral-medium-latest",
    "Mistral-Large": "mistral-large-latest",
}

MAX_WORKERS = 3
MAX_RETRIES = 6
BASE_BACKOFF = 0.8
JITTER = 0.25
REQUEST_SPACING = 0.01

SEED = 42
random.seed(SEED)
DETERMINISTIC_JITTER = True
STORE_VOTES_LIST = True

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# ======================================================
# 1) LOAD API
# ======================================================
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("Missing MISTRAL_API_KEY in environment (.env).")

client = Mistral(api_key=api_key)


# ======================================================
# 2) PATHS
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
# 2.1) BASIC DATA HYGIENE
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
# 3) MINIMAL ARABIC NORMALIZATION
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
# 4) PROMPT
# ======================================================
SYSTEM_MSG = (
    "You are a strict emotion classifier for Arabic text. "
    "Return ONLY valid JSON with exactly one key: label. "
    "The value MUST be one of: joy, anger, sadness, fear, neutral. "
    "Use lowercase only. "
    "If unsure, output neutral. "
    "Do not add explanations or extra text."
)

PROMPT = """
Choose exactly ONE label from: joy, anger, sadness, fear, neutral.
Return ONLY JSON like: {"label":"joy"} (lowercase only).
If unclear, return {"label":"neutral"}.

Sentence: "{TEXT}"
""".strip()


# ======================================================
# 5) PARSING HELPERS
# ======================================================
def clean_label(lbl) -> str:
    s = str(lbl).strip().lower()
    first = re.split(r"\s+", s)[0]

    if first in ALLOWED_LABELS:
        return first

    for lab in ALLOWED_LABELS:
        if re.search(rf"\b{re.escape(lab)}\b", s):
            return lab

    return UNKNOWN_LABEL


def parse_json_label(raw_text: str) -> str:
    raw = str(raw_text).strip()

    match = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "label" in obj:
                return clean_label(obj["label"])
        except Exception:
            pass

    return clean_label(raw)


# ======================================================
# 6) BACKOFF HELPERS
# ======================================================
def _is_transient_error(msg_lower: str) -> bool:
    return (
        ("rate" in msg_lower)
        or ("429" in msg_lower)
        or ("too many requests" in msg_lower)
        or ("overloaded" in msg_lower)
        or ("temporarily" in msg_lower and "unavailable" in msg_lower)
        or ("timeout" in msg_lower)
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
# 7) MISTRAL RESPONSE EXTRACTION
# ======================================================
def extract_chat_text(resp) -> str:
    try:
        content = resp.choices[0].message.content
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if txt:
                        parts.append(txt)
                else:
                    txt = getattr(item, "text", None)
                    if txt:
                        parts.append(txt)
            return "".join(parts).strip()
    except Exception:
        pass

    return ""


# ======================================================
# 8) CALL MODEL
# ======================================================
def call_model(model_id: str, model_name: str, text: str, temperature: float, sample_i=None, vote_i=None):
    prompt = PROMPT.replace("{TEXT}", text)
    salt = f"{PROVIDER_NAME}|{model_name}|{sample_i}|{vote_i}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if REQUEST_SPACING > 0:
                time.sleep(REQUEST_SPACING)

            resp = client.chat.complete(
                model=model_id,
                temperature=float(temperature),
                max_tokens=20,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
            )

            raw = extract_chat_text(resp)
            if not raw:
                return UNKNOWN_LABEL, 1

            return raw, 0

        except Exception as e:
            err = str(e).lower()
            if _is_transient_error(err):
                _sleep_backoff(attempt, salt)
                continue
            return UNKNOWN_LABEL, 1

    return UNKNOWN_LABEL, 1


# ======================================================
# 9) VOTING HELPERS
# ======================================================
def _vote_pattern(counter: Counter) -> str:
    if not counter:
        return "0"
    return "-".join(map(str, sorted(counter.values(), reverse=True)))


def classify_one(model_id: str, model_name: str, text: str, votes: int = VOTES, sample_i=None):
    preds = []
    err_calls = 0
    calls_made = 0

    for v in range(votes):
        raw, err = call_model(
            model_id=model_id,
            model_name=model_name,
            text=text,
            temperature=TEMP_VOTES,
            sample_i=sample_i,
            vote_i=v,
        )
        calls_made += 1
        err_calls += err
        preds.append(parse_json_label(raw))

    counts_all = Counter(preds)
    unknown_votes = counts_all.get(UNKNOWN_LABEL, 0)

    valid = [p for p in preds if p != UNKNOWN_LABEL]
    counts_valid = Counter(valid)
    pattern = _vote_pattern(counts_valid)

    if not valid:
        return UNKNOWN_LABEL, err_calls, 1, calls_made, preds, pattern, unknown_votes, "all_failed"

    # joy / anger / sadness
    if len(valid) == votes and len(counts_valid) == votes:
        return UNKNOWN_LABEL, err_calls, 1, calls_made, preds, pattern, unknown_votes, "all_different_111"

    # joy / joy / anger
    top_label, top_count = counts_valid.most_common(1)[0]
    if top_count >= 2:
        return top_label, err_calls, 0, calls_made, preds, pattern, unknown_votes, "majority"

    # joy / anger / unknown
    return UNKNOWN_LABEL, err_calls, 1, calls_made, preds, pattern, unknown_votes, "no_majority_11u"


# ======================================================
# 10) RUN BENCHMARK PER MODEL
# ======================================================
summary_rows = []

for MODEL_NAME, MODEL_ID in MODELS.items():
    print(f"\nRunning {MODEL_NAME} ...")

    n = len(df)

    preds = [None] * n
    abstained_arr = [0] * n
    vote_patterns = [None] * n
    unknown_votes_arr = [0] * n
    unknown_reasons = [None] * n
    calls_made_arr = [0] * n
    votes_list_arr = [None] * n

    error_calls_total = 0
    abstain_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_one, MODEL_ID, MODEL_NAME, txt, VOTES, i): i
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

            error_calls_total += err_calls
            abstain_count += abst_inc

    y_true = df["emotion"].tolist()
    y_pred = preds

    unknown_total = y_pred.count(UNKNOWN_LABEL)
    unknown_rate = unknown_total / len(y_pred)
    coverage = 1.0 - unknown_rate

    pd.Series(unknown_reasons).value_counts(dropna=False).to_csv(
        os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__UNKNOWN_REASON_DISTRIBUTION.csv"),
        encoding="utf-8-sig"
    )

    # ======================================================
    # 11) STRICT METRICS
    # ======================================================
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
    for i in range(len(STRICT_LABELS)):
        for j in range(len(STRICT_LABELS)):
            plt.text(j, i, str(cm_strict_unknown[i, j]), ha="center", va="center")
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
    for i in range(len(ALLOWED_LABELS)):
        for j in range(len(ALLOWED_LABELS)):
            plt.text(j, i, str(cm_strict_5[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_STRICT_5LABELS.png"),
        dpi=300
    )
    plt.close()

    # ======================================================
    # 12) COVERED METRICS
    # ======================================================
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

        report_cov = classification_report(
            y_true_cov,
            y_pred_cov,
            labels=COVERED_LABELS,
            output_dict=True,
            zero_division=0
        )
        pd.DataFrame(report_cov).transpose().to_csv(
            os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__REPORT_COVERED.csv"),
            encoding="utf-8-sig"
        )

        cm_cov = confusion_matrix(y_true_cov, y_pred_cov, labels=COVERED_LABELS)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_cov, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(COVERED_LABELS)), COVERED_LABELS, rotation=45)
        plt.yticks(range(len(COVERED_LABELS)), COVERED_LABELS)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{MODEL_NAME} Confusion Matrix (COVERED)")
        for i in range(len(COVERED_LABELS)):
            for j in range(len(COVERED_LABELS)):
                plt.text(j, i, str(cm_cov[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__CM_COVERED.png"),
            dpi=300
        )
        plt.close()
    else:
        covered_accuracy = 0.0
        covered_macro_f1 = 0.0
        covered_micro_f1 = 0.0
        covered_weighted_f1 = 0.0

    # ======================================================
    # 13) SAVE PREDICTIONS
    # ======================================================
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
    })

    if STORE_VOTES_LIST:
        out_df["votes_list"] = votes_list_arr

    out_df.to_csv(
        os.path.join(PRED_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__{MODEL_NAME}__PREDICTIONS.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    reason_counter = Counter(unknown_reasons)

    summary_rows.append({
        "provider": PROVIDER_NAME,
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "setup": "zero-shot + three-vote self-consistency (T=0.3) + valid-vote aggregation",
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

        "covered_accuracy": covered_accuracy,
        "covered_macro_f1": covered_macro_f1,
        "covered_micro_f1": covered_micro_f1,
        "covered_weighted_f1": covered_weighted_f1,

        "unknown_reason_all_failed": int(reason_counter.get("all_failed", 0)),
        "unknown_reason_all_different_111": int(reason_counter.get("all_different_111", 0)),
        "unknown_reason_no_majority_11u": int(reason_counter.get("no_majority_11u", 0)),
    })


# ======================================================
# 14) SAVE SUMMARY
# ======================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(METRIC_DIR, f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_Q1_strong.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nMistral three-vote benchmark completed successfully for all manuscript models.")
print("RUN_ID:", RUN_ID)
print("Models run:", list(MODELS.keys()))