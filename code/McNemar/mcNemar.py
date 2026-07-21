# mcNemar_final_12_models.py
# ============================================================
# ZERO-SHOT SINGLE-PASS MCNEMAR ANALYSIS
# FINAL 12 MANUSCRIPT MODELS
#
# This analysis script:
#   1) Scans supplied CSV prediction files recursively
#   2) Detects gold, prediction, and model columns
#   3) Excludes few-shot and three-vote result files
#   4) Selects the exact manuscript run listed in OFFICIAL_FILES
#   5) Runs all 66 pairwise exact McNemar tests
#   6) Applies Holm correction across all comparisons
#   7) Produces CSV, LaTeX, and heatmap outputs
#
# Important:
# The original sample-level prediction files are not included
# in the public repository. To reproduce the manuscript analysis,
# place the authorised files listed in OFFICIAL_FILES under
# output/one_vote/ while preserving their filenames.
# ============================================================

from __future__ import annotations

import math
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Expected location:
# repository_root/code/McNemar/mcNemar_final_12_models.py
PROJECT_ROOT = BASE_DIR.parents[1]

PREDICTIONS_DIR = PROJECT_ROOT / "output" / "one_vote"
OUTPUT_DIR = PROJECT_ROOT / "output" / "mcnemar_12_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_SAMPLE_COUNT = 1002
ALPHA = 0.05

UNKNOWN_VALUES = {
    "unknown",
    "__unknown__",
    "unk",
    "nan",
    "",
    None,
}

# Use exactly the authorised files that generated the manuscript results.
# Provider subfolders are allowed because files are resolved recursively.
OFFICIAL_FILES = {
    "OpenAI-o3": "OpenAI__RUN20260305_162635__V1__OpenAI-o3__PREDICTIONS.csv",
    "GPT-4o": "OpenAI__RUN20260305_095940__V1__GPT-4o__PREDICTIONS.csv",
    "GPT-5": "OpenAI__RUN20260305_162635__V1__GPT-5__PREDICTIONS.csv",
    "Gemini-2.5-Flash": "Google__RUN20260305_134206__V1__Gemini-2.5-Flash__PREDICTIONS.csv",
    "Gemma-3-12B": "Google__RUN20260306_160835__V1__Gemma-3-12B__PREDICTIONS.csv",
    "Claude-Sonnet-4.6": "Claude__RUN20260305_101527__V1__Claude-Sonnet-4.6__PREDICTIONS.csv",
    "GPT-4.1-mini": "OpenAI__RUN20260305_095940__V1__GPT-4.1-mini__PREDICTIONS.csv",
    "Qwen-3-32B": "Groq__Qwen-3-32B__RUN20260322_152314__V1__predictions_V1.csv",
    "Allam-2-7B": "Groq__RUN20260304_141138__V1__Allam-2-7B__PREDICTIONS.csv",
    "Mistral-Medium": "Mistral__RUN20260305_143021__V1__mistral-medium-latest__PREDICTIONS.csv",
    "Mistral-Large": "Mistral__RUN20260318_134239__MISTRAL_LARGE_FAST__mistral-large-latest__PREDICTIONS.csv",
    "LLaMA-3.1-8B": "Groq__RUN20260322_134354__V1__Groq-LLaMA-3.1-8B__PREDICTIONS.csv",
}

MODEL_ORDER = [
    "OpenAI-o3",
    "GPT-4o",
    "GPT-5",
    "Gemini-2.5-Flash",
    "Gemma-3-12B",
    "Claude-Sonnet-4.6",
    "GPT-4.1-mini",
    "Qwen-3-32B",
    "Allam-2-7B",
    "Mistral-Medium",
    "Mistral-Large",
    "LLaMA-3.1-8B",
]

KEEP_MODELS = set(MODEL_ORDER)
EXCLUDE_MODELS: set[str] = set()

if set(OFFICIAL_FILES) != KEEP_MODELS:
    missing = sorted(KEEP_MODELS - set(OFFICIAL_FILES))
    unexpected = sorted(set(OFFICIAL_FILES) - KEEP_MODELS)
    raise RuntimeError(
        "OFFICIAL_FILES must match MODEL_ORDER exactly. "
        f"Missing entries: {missing}; unexpected entries: {unexpected}."
    )

EXCLUDE_FILE_PATTERNS = [
    r"fewshot",
    r"few_shot",
    r"few-shot",
    r"three[_-]?vote",
    r"self[_-]?consistency",
    r"__v3__",
    r"summary",
    r"report",
    r"distribution",
    r"mcnemar",
]

SELECTED_MAIN_COMPARISONS = [
    ("OpenAI-o3", "GPT-4o"),
    ("OpenAI-o3", "GPT-5"),
    ("GPT-4o", "GPT-5"),
    ("OpenAI-o3", "Gemini-2.5-Flash"),
    ("GPT-4o", "Gemini-2.5-Flash"),
    ("GPT-5", "Gemini-2.5-Flash"),
]


# ============================================================
# NORMALISATION
# ============================================================

UNKNOWN_VALUES_LOWER = {
    str(value).strip().lower()
    for value in UNKNOWN_VALUES
    if value is not None
}


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    if text in UNKNOWN_VALUES_LOWER:
        return "unknown"

    aliases = {
        "angry": "anger",
        "mad": "anger",
        "happy": "joy",
        "hapy": "joy",
        "sad": "sadness",
    }
    return aliases.get(text, text)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def canonical_model_name(value: object) -> str:
    raw = str(value).strip()
    lower = raw.lower()
    compact = compact_name(raw)

    exact_aliases = {
        "gpt-4o": "GPT-4o",
        "openai-o3": "OpenAI-o3",
        "o3": "OpenAI-o3",
        "gpt-5": "GPT-5",
        "gemini-2.5-flash": "Gemini-2.5-Flash",
        "models/gemini-2.5-flash": "Gemini-2.5-Flash",
        "gemini 2.5 flash": "Gemini-2.5-Flash",
        "gemma-3-12b": "Gemma-3-12B",
        "models/gemma-3-12b-it": "Gemma-3-12B",
        "claude-sonnet-4.6": "Claude-Sonnet-4.6",
        "gpt-4.1-mini": "GPT-4.1-mini",
        "qwen-3-32b": "Qwen-3-32B",
        "qwen3-32b": "Qwen-3-32B",
        "qwen/qwen3-32b": "Qwen-3-32B",
        "qwen/qwen3-32b-instruct": "Qwen-3-32B",
        "allam-2-7b": "Allam-2-7B",
        "allam-2-7b-instruct": "Allam-2-7B",
        "mistral-medium": "Mistral-Medium",
        "mistral-medium-latest": "Mistral-Medium",
        "mistral-large": "Mistral-Large",
        "mistral-large-latest": "Mistral-Large",
        "llama-3.1-8b": "LLaMA-3.1-8B",
        "llama-3.1-8b-instant": "LLaMA-3.1-8B",
        "groq-llama-3.1-8b": "LLaMA-3.1-8B",
        "meta-llama/llama-3.1-8b-instruct": "LLaMA-3.1-8B",
        "meta-llama/llama-3.1-8b-instruct-turbo": "LLaMA-3.1-8B",
    }

    if lower in exact_aliases:
        return exact_aliases[lower]

    compact_patterns = [
        ("gemini25flash", "Gemini-2.5-Flash"),
        ("gemma312b", "Gemma-3-12B"),
        ("claudesonnet46", "Claude-Sonnet-4.6"),
        ("gpt41mini", "GPT-4.1-mini"),
        ("gpt4o", "GPT-4o"),
        ("openaio3", "OpenAI-o3"),
        ("gpt5", "GPT-5"),
        ("qwen332b", "Qwen-3-32B"),
        ("allam27b", "Allam-2-7B"),
        ("mistralmediumlatest", "Mistral-Medium"),
        ("mistralmedium", "Mistral-Medium"),
        ("mistrallargelatest", "Mistral-Large"),
        ("mistrallarge", "Mistral-Large"),
        ("groqllama318b", "LLaMA-3.1-8B"),
        ("llama318binstant", "LLaMA-3.1-8B"),
        ("llama318b", "LLaMA-3.1-8B"),
    ]

    for pattern, canonical in compact_patterns:
        if pattern in compact:
            return canonical

    return raw


# ============================================================
# COLUMN / FILE DETECTION
# ============================================================


def detect_column(frame: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def find_gold_and_pred_columns(frame: pd.DataFrame) -> Tuple[str, str]:
    gold_candidates = [
        "gold_label",
        "true_label",
        "label",
        "gold",
        "ground_truth",
        "y_true",
        "target",
        "true",
        "goldlabel",
        "emotion",
    ]

    prediction_candidates = [
        "predicted_label",
        "prediction",
        "pred_label",
        "model_label",
        "final_label",
        "label_pred",
        "y_pred",
        "pred",
        "predicted",
        "output_label",
        "model_prediction",
        "prediction_label",
    ]

    gold_column = detect_column(frame, gold_candidates)
    prediction_column = detect_column(frame, prediction_candidates)

    if gold_column is None:
        raise ValueError(
            "Could not detect gold-label column. "
            f"Available columns: {list(frame.columns)}"
        )

    if prediction_column is None:
        raise ValueError(
            "Could not detect prediction column. "
            f"Available columns: {list(frame.columns)}"
        )

    if gold_column == prediction_column:
        raise ValueError(
            "Gold and prediction columns resolved to the same column: "
            f"{gold_column}. Check the prediction-file schema."
        )

    return gold_column, prediction_column


def detect_model_name(frame: pd.DataFrame, filepath: Path) -> str:
    for column in ("model_name", "model", "model_id"):
        if column in frame.columns:
            values = (
                frame[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            for value in values:
                canonical = canonical_model_name(value)
                if canonical in KEEP_MODELS:
                    return canonical

    return canonical_model_name(filepath.stem)


def file_should_be_excluded(filepath: Path) -> bool:
    filename = filepath.name.lower()
    return any(
        re.search(pattern, filename, flags=re.IGNORECASE)
        for pattern in EXCLUDE_FILE_PATTERNS
    )


def is_zero_shot_single_pass(frame: pd.DataFrame, filepath: Path) -> bool:
    if file_should_be_excluded(filepath):
        return False

    setup_column = detect_column(frame, ["setup", "experiment_setup", "mode"])
    if setup_column is not None:
        setup_text = " ".join(
            frame[setup_column]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )
        if any(
            marker in setup_text
            for marker in (
                "few-shot",
                "few shot",
                "fewshot",
                "three-vote",
                "three vote",
                "self-consistency",
                "self consistency",
            )
        ):
            return False

    votes_column = detect_column(frame, ["votes", "n_votes"])
    if votes_column is not None:
        votes = pd.to_numeric(frame[votes_column], errors="coerce").dropna().unique()
        if len(votes) and not np.all(votes == 1):
            return False

    temperature_column = detect_column(frame, ["temp_votes", "temperature", "temp"])
    if temperature_column is not None:
        temperatures = (
            pd.to_numeric(frame[temperature_column], errors="coerce")
            .dropna()
            .unique()
        )
        if len(temperatures) and not np.allclose(temperatures, 0.0):
            return False

    return True


def read_csv_robust(filepath: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding="utf-8")


# ============================================================
# METRICS / MCNEMAR
# ============================================================


def strict_accuracy(gold: pd.Series, prediction: pd.Series) -> float:
    return float((gold == prediction).astype(int).mean())


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0

    return float(
        binomtest(
            k=b,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def format_p_for_latex(value: float) -> str:
    return r"$<0.001$" if value < 0.001 else f"{value:.3f}"


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ============================================================
# LOAD ALL SUPPLIED CSV FILES
# ============================================================

if not PREDICTIONS_DIR.exists():
    raise FileNotFoundError(
        "Predictions folder does not exist:\n"
        f"{PREDICTIONS_DIR}\n\n"
        "Create the folder and place the authorised primary "
        "zero-shot prediction files under it."
    )

all_files = sorted(PREDICTIONS_DIR.rglob("*.csv"))

if not all_files:
    raise FileNotFoundError(
        "No CSV files were found recursively under:\n"
        f"{PREDICTIONS_DIR}\n\n"
        "The original sample-level outputs are not distributed "
        "in the public repository. Supply the authorised files "
        "before running this analysis."
    )

file_rows: list[dict] = []
usable_files: list[dict] = []

for filepath in all_files:
    if file_should_be_excluded(filepath):
        continue

    try:
        frame = read_csv_robust(filepath)

        if not is_zero_shot_single_pass(frame, filepath):
            continue

        gold_column, prediction_column = find_gold_and_pred_columns(frame)
        model_name = detect_model_name(frame, filepath)

        if model_name not in KEEP_MODELS or model_name in EXCLUDE_MODELS:
            continue

        gold = frame[gold_column].map(normalize_label)
        prediction = frame[prediction_column].map(normalize_label)
        accuracy = strict_accuracy(gold, prediction)

        file_rows.append(
            {
                "file": filepath.name,
                "filepath": str(filepath),
                "model": model_name,
                "gold_col": gold_column,
                "pred_col": prediction_column,
                "n_rows": len(frame),
                "strict_accuracy": accuracy,
            }
        )

        usable_files.append(
            {
                "filepath": filepath,
                "file": filepath.name,
                "model": model_name,
                "df": frame,
                "gold_col": gold_column,
                "pred_col": prediction_column,
                "strict_accuracy": accuracy,
            }
        )

    except Exception as error:
        print(f"[SKIP] {filepath.name} -> {type(error).__name__}: {error}")

if not usable_files:
    raise RuntimeError(
        "No usable primary zero-shot single-pass prediction files were found."
    )

all_detected_df = pd.DataFrame(file_rows).sort_values(
    ["strict_accuracy", "model", "file"],
    ascending=[False, True, True],
)

all_detected_df.to_csv(
    OUTPUT_DIR / "all_zero_shot_files_detected.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SELECT OFFICIAL MANUSCRIPT FILES
# ============================================================

usable_by_filename: Dict[str, list[dict]] = {}
for item in usable_files:
    usable_by_filename.setdefault(item["file"], []).append(item)

best_by_model: Dict[str, dict] = {}
manifest_rows: list[dict] = []

for model in MODEL_ORDER:
    expected_filename = OFFICIAL_FILES[model]
    matches = usable_by_filename.get(expected_filename, [])

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Official prediction file for {model} was not found recursively under:\n"
            f"{PREDICTIONS_DIR}\n\n"
            f"Expected filename:\n{expected_filename}\n\n"
            "The original prediction files are not distributed in the public "
            "repository. Place the authorised zero-shot prediction file under "
            "the configured PREDICTIONS_DIR and preserve its original filename."
        )

    if len(matches) > 1:
        locations = "\n".join(f" - {item['filepath']}" for item in matches)
        raise RuntimeError(
            f"More than one file named {expected_filename} was found for {model}:\n"
            f"{locations}\n"
            "Keep exactly one authorised copy of the official prediction file."
        )

    item = matches[0]

    if item["model"] != model:
        raise RuntimeError(
            "Official file/model mismatch:\n"
            f"Expected model: {model}\n"
            f"Detected model: {item['model']}\n"
            f"File: {item['filepath']}"
        )

    best_by_model[model] = item
    manifest_rows.append(
        {
            "model": model,
            "official_filename": expected_filename,
            "filepath": str(item["filepath"]),
            "gold_col": item["gold_col"],
            "pred_col": item["pred_col"],
            "strict_accuracy": item["strict_accuracy"],
            "n_rows": len(item["df"]),
        }
    )

selected_df = pd.DataFrame(manifest_rows)
selected_df.to_csv(
    OUTPUT_DIR / "selected_official_run_per_model.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 100)
print("SELECTED OFFICIAL ZERO-SHOT FILE PER MODEL")
print("=" * 100)
print(selected_df.to_string(index=False))


# ============================================================
# BUILD ALIGNED MODEL DATA
# ============================================================

model_data: Dict[str, dict] = {}

for model in MODEL_ORDER:
    item = best_by_model[model]
    frame = item["df"].copy()

    if len(frame) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"{model}: expected {EXPECTED_SAMPLE_COUNT} rows, "
            f"found {len(frame)} in {item['file']}."
        )

    gold = frame[item["gold_col"]].map(normalize_label)
    prediction = frame[item["pred_col"]].map(normalize_label)

    text_column = detect_column(
        frame,
        ["text", "sentence", "input_text", "arabic_text"],
    )
    sample_id_column = detect_column(
        frame,
        ["sample_id", "dataset_index", "row_id", "id", "index"],
    )

    # Use a unique sample ID when available. Otherwise preserve row order.
    # Normalised text is only a cross-file verification field.
    if sample_id_column is not None:
        candidate_key = frame[sample_id_column].astype(str).str.strip()
        if candidate_key.eq("").any() or candidate_key.duplicated().any():
            print(
                f"[WARNING] {model}: column '{sample_id_column}' contains "
                "blank or duplicate values; falling back to row order."
            )
            sample_key = pd.Series(range(len(frame)), index=frame.index).astype(str)
        else:
            sample_key = candidate_key
    else:
        sample_key = pd.Series(range(len(frame)), index=frame.index).astype(str)

    if text_column is not None:
        text_check = frame[text_column].map(normalize_text)
    else:
        text_check = pd.Series([""] * len(frame), index=frame.index)

    aligned = pd.DataFrame(
        {
            "sample_key": sample_key,
            "text_check": text_check,
            "gold": gold,
            "pred": prediction,
        }
    )

    if aligned["sample_key"].duplicated().any():
        raise ValueError(
            f"{model}: duplicated sample IDs or row keys in {item['file']}."
        )

    aligned["correct"] = (aligned["gold"] == aligned["pred"]).astype(int)

    model_data[model] = {
        "file": item["file"],
        "strict_accuracy": item["strict_accuracy"],
        "data": aligned,
    }


# ============================================================
# CROSS-MODEL ALIGNMENT
# ============================================================

base_model = MODEL_ORDER[0]
merged = model_data[base_model]["data"].rename(
    columns={
        "text_check": f"{base_model}__text",
        "gold": f"{base_model}__gold",
        "pred": f"{base_model}__pred",
        "correct": f"{base_model}__correct",
    }
)

for model in MODEL_ORDER[1:]:
    current = model_data[model]["data"].rename(
        columns={
            "text_check": f"{model}__text",
            "gold": f"{model}__gold",
            "pred": f"{model}__pred",
            "correct": f"{model}__correct",
        }
    )
    merged = merged.merge(
        current,
        on="sample_key",
        how="inner",
        validate="one_to_one",
    )

if len(merged) != EXPECTED_SAMPLE_COUNT:
    raise ValueError(
        "Model files are not aligned. "
        f"Expected {EXPECTED_SAMPLE_COUNT} common samples, found {len(merged)}."
    )

base_gold = merged[f"{base_model}__gold"]
base_text = merged[f"{base_model}__text"]

for model in MODEL_ORDER[1:]:
    current_gold = merged[f"{model}__gold"]
    current_text = merged[f"{model}__text"]

    if not base_gold.equals(current_gold):
        mismatch_count = int((base_gold != current_gold).sum())
        raise ValueError(
            "Gold labels are not aligned between "
            f"{base_model} and {model}. Mismatches: {mismatch_count}. "
            "Check sample IDs and dataset row order."
        )

    comparable_text = base_text.ne("") & current_text.ne("")
    text_mismatch = comparable_text & base_text.ne(current_text)

    if text_mismatch.any():
        mismatch_count = int(text_mismatch.sum())
        raise ValueError(
            "Input texts are not aligned between "
            f"{base_model} and {model}. Mismatches: {mismatch_count}. "
            "Check sample ordering or sample IDs."
        )


# ============================================================
# MCNEMAR — ALL 66 PAIRS
# ============================================================

results: list[dict] = []

for model_a, model_b in combinations(MODEL_ORDER, 2):
    correct_a = merged[f"{model_a}__correct"].astype(int)
    correct_b = merged[f"{model_b}__correct"].astype(int)

    both_correct = int(((correct_a == 1) & (correct_b == 1)).sum())
    a_correct_b_wrong = int(((correct_a == 1) & (correct_b == 0)).sum())
    a_wrong_b_correct = int(((correct_a == 0) & (correct_b == 1)).sum())
    both_wrong = int(((correct_a == 0) & (correct_b == 0)).sum())

    raw_p_value = exact_mcnemar_pvalue(
        a_correct_b_wrong,
        a_wrong_b_correct,
    )

    results.append(
        {
            "model_a": model_a,
            "acc_a": model_data[model_a]["strict_accuracy"],
            "model_b": model_b,
            "acc_b": model_data[model_b]["strict_accuracy"],
            "both_correct": both_correct,
            "a_correct_b_wrong": a_correct_b_wrong,
            "a_wrong_b_correct": a_wrong_b_correct,
            "both_wrong": both_wrong,
            "discordant_total": a_correct_b_wrong + a_wrong_b_correct,
            "raw_p_value": raw_p_value,
            "file_a": model_data[model_a]["file"],
            "file_b": model_data[model_b]["file"],
        }
    )

results_df = pd.DataFrame(results)
expected_comparisons = math.comb(len(MODEL_ORDER), 2)

if expected_comparisons != 66:
    raise RuntimeError(
        "Final manuscript configuration must produce 66 comparisons, "
        f"but MODEL_ORDER currently produces {expected_comparisons}."
    )

if len(results_df) != expected_comparisons:
    raise RuntimeError(
        f"Expected {expected_comparisons} comparisons, computed {len(results_df)}."
    )

reject, adjusted_p_values, _, _ = multipletests(
    results_df["raw_p_value"].to_numpy(),
    alpha=ALPHA,
    method="holm",
)

results_df["holm_adjusted_p_value"] = adjusted_p_values
results_df["significant_holm_0.05"] = np.where(reject, "Yes", "No")
results_df["significance_label"] = np.where(
    reject,
    "Significant",
    "Not significant",
)

results_df.to_csv(
    OUTPUT_DIR / "mcnemar_all_pairs_detailed_66.csv",
    index=False,
    encoding="utf-8-sig",
)

summary_df = results_df[
    [
        "model_a",
        "acc_a",
        "model_b",
        "acc_b",
        "a_correct_b_wrong",
        "a_wrong_b_correct",
        "raw_p_value",
        "holm_adjusted_p_value",
        "significant_holm_0.05",
    ]
].copy()

summary_df.to_csv(
    OUTPUT_DIR / "mcnemar_all_pairs_summary_66.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SELECTED MAIN-PAPER TABLE
# ============================================================

selected_keys = {frozenset(pair) for pair in SELECTED_MAIN_COMPARISONS}

selected_main_df = results_df[
    results_df.apply(
        lambda row: frozenset((row["model_a"], row["model_b"])) in selected_keys,
        axis=1,
    )
].copy()

selected_order = {
    frozenset(pair): index
    for index, pair in enumerate(SELECTED_MAIN_COMPARISONS)
}

selected_main_df["__order"] = selected_main_df.apply(
    lambda row: selected_order[frozenset((row["model_a"], row["model_b"]))],
    axis=1,
)

selected_main_df = (
    selected_main_df
    .sort_values("__order")
    .drop(columns="__order")
)

selected_main_df.to_csv(
    OUTPUT_DIR / "selected_mcnemar_main_table.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# LATEX — MAIN TABLE
# ============================================================

main_latex_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    (
        r"\caption{Selected pairwise exact McNemar test results "
        r"for the zero-shot single-pass benchmark. The reported "
        r"$p$-values were adjusted across all 66 pairwise "
        r"comparisons using the Holm step-down procedure. "
        r"Statistical significance was determined using "
        r"Holm-adjusted $p$-values at $\alpha=0.05$.}"
    ),
    r"\label{tab:mcnemar_main}",
    r"\begin{tabular}{llcccc}",
    r"\toprule",
    (
        r"Model A & Model B & $b$ & $c$ & "
        r"Holm-adjusted $p$-value & Significance \\"
    ),
    r"\midrule",
]

for _, row in selected_main_df.iterrows():
    significance = (
        r"\textbf{Significant}"
        if row["significance_label"] == "Significant"
        else "Not significant"
    )

    main_latex_lines.append(
        f"{latex_escape(row['model_a'])} & "
        f"{latex_escape(row['model_b'])} & "
        f"{int(row['a_correct_b_wrong'])} & "
        f"{int(row['a_wrong_b_correct'])} & "
        f"{format_p_for_latex(float(row['holm_adjusted_p_value']))} & "
        f"{significance} \\\\"
    )

main_latex_lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
)

(OUTPUT_DIR / "selected_mcnemar_main_table.tex").write_text(
    "\n".join(main_latex_lines),
    encoding="utf-8",
)


# ============================================================
# LATEX — FULL APPENDIX
# ============================================================

appendix_lines = [
    r"\begin{landscape}",
    r"\setlength{\LTleft}{\fill}",
    r"\setlength{\LTright}{\fill}",
    "",
    r"\begin{longtable}{p{3cm} p{3cm} c c c c}",
    (
        r"\caption{Complete pairwise exact McNemar test results "
        r"for all 12 evaluated models under the zero-shot "
        r"single-pass setting. Reported $p$-values were adjusted "
        r"across all 66 comparisons using the Holm step-down procedure.}"
    ),
    r"\label{tab:mcnemar_full} \\",
    r"\toprule",
    (
        r"Model A & Model B & $b$ & $c$ & "
        r"Adjusted $p$-value & Significance \\"
    ),
    r"\midrule",
    r"\endfirsthead",
    "",
    r"\toprule",
    (
        r"Model A & Model B & $b$ & $c$ & "
        r"Adjusted $p$-value & Significance \\"
    ),
    r"\midrule",
    r"\endhead",
    "",
    r"\bottomrule",
    r"\endfoot",
    "",
]

for _, row in results_df.iterrows():
    appendix_lines.append(
        f"{latex_escape(row['model_a'])} & "
        f"{latex_escape(row['model_b'])} & "
        f"{int(row['a_correct_b_wrong'])} & "
        f"{int(row['a_wrong_b_correct'])} & "
        f"{format_p_for_latex(float(row['holm_adjusted_p_value']))} & "
        f"{row['significance_label']} \\\\"
    )

appendix_lines.extend(
    [
        "",
        r"\end{longtable}",
        r"\end{landscape}",
        "",
    ]
)

(OUTPUT_DIR / "appendix_mcnemar_66.tex").write_text(
    "\n".join(appendix_lines),
    encoding="utf-8",
)


# ============================================================
# HEATMAP
# ============================================================

model_to_index = {
    model: index
    for index, model in enumerate(MODEL_ORDER)
}

matrix = np.ones(
    (len(MODEL_ORDER), len(MODEL_ORDER)),
    dtype=float,
)

for _, row in results_df.iterrows():
    i = model_to_index[row["model_a"]]
    j = model_to_index[row["model_b"]]
    value = float(row["holm_adjusted_p_value"])
    matrix[i, j] = value
    matrix[j, i] = value

np.fill_diagonal(matrix, np.nan)

figure, axis = plt.subplots(
    figsize=(14, 12),
    constrained_layout=True,
)

image = axis.imshow(
    matrix,
    vmin=0.0,
    vmax=1.0,
)

axis.set_xticks(range(len(MODEL_ORDER)))
axis.set_yticks(range(len(MODEL_ORDER)))
axis.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
axis.set_yticklabels(MODEL_ORDER)

axis.set_title(
    "Holm-adjusted exact McNemar p-values\n"
    "Zero-shot single-pass benchmark (12 models, 66 comparisons)"
)

for i in range(len(MODEL_ORDER)):
    for j in range(len(MODEL_ORDER)):
        if i == j:
            continue

        p_value = matrix[i, j]
        display_value = "<.001" if p_value < 0.001 else f"{p_value:.3f}"

        axis.text(
            j,
            i,
            display_value,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold" if p_value < ALPHA else "normal",
        )

colorbar = figure.colorbar(image, ax=axis)
colorbar.set_label("Holm-adjusted p-value")

figure.savefig(
    OUTPUT_DIR / "McNemar_HeatMap_12_models.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 100)
print("MCNEMAR COMPLETE — 12 MODELS / 66 COMPARISONS")
print("=" * 100)

print(
    selected_main_df[
        [
            "model_a",
            "model_b",
            "a_correct_b_wrong",
            "a_wrong_b_correct",
            "holm_adjusted_p_value",
            "significance_label",
        ]
    ].to_string(index=False)
)

print("\nSaved files:")
for filename in (
    "all_zero_shot_files_detected.csv",
    "selected_official_run_per_model.csv",
    "mcnemar_all_pairs_detailed_66.csv",
    "mcnemar_all_pairs_summary_66.csv",
    "selected_mcnemar_main_table.csv",
    "selected_mcnemar_main_table.tex",
    "appendix_mcnemar_66.tex",
    "McNemar_HeatMap_12_models.png",
):
    print(f"- {OUTPUT_DIR / filename}")
