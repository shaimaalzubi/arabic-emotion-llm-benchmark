#!/usr/bin/env python3
"""
Bootstrap confidence intervals for classification metrics.

This script computes 95% percentile bootstrap confidence intervals for:
- Accuracy
- Macro-F1

The bootstrap is performed at the evaluation-sample level by resampling
paired (reference label, model prediction) observations with replacement.

Strict-evaluation handling:
- Predictions equal to "unknown" remain in the resampled data.
- They are treated as incorrect predictions.
- Macro-F1 is computed over the valid emotion labels only.

Example:
    python bootstrap_ci.py \
        --input predictions.csv \
        --model GPT-4o \
        --setting zero-shot \
        --true-col gold_label \
        --pred-col prediction \
        --output bootstrap_ci_gpt4o_zero_shot.csv

Dependencies:
    pandas
    numpy
    scikit-learn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


DEFAULT_LABELS = ("anger", "joy", "sadness", "fear", "neutral")
TRUE_COLUMN_CANDIDATES = (
    "gold_label",
    "gold",
    "true_label",
    "label",
    "emotion",
    "reference_label",
    "y_true",
)
PRED_COLUMN_CANDIDATES = (
    "prediction",
    "pred",
    "predicted_label",
    "model_prediction",
    "final_prediction",
    "y_pred",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute percentile bootstrap confidence intervals for "
            "accuracy and macro-F1."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV file containing reference and predicted labels.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV file for the summary results.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to store in the output.",
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=("zero-shot", "few-shot", "three-vote"),
        help="Evaluation setting.",
    )
    parser.add_argument(
        "--true-col",
        default=None,
        help=(
            "Reference-label column. If omitted, the script attempts "
            "automatic detection."
        ),
    )
    parser.add_argument(
        "--pred-col",
        default=None,
        help=(
            "Prediction column. If omitted, the script attempts "
            "automatic detection."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Valid class labels used for macro-F1.",
    )
    parser.add_argument(
        "--unknown-label",
        default="unknown",
        help='Label used for non-evaluable predictions. Default: "unknown".',
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples. Default: 10000.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level between 0 and 1. Default: 0.95.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=1002,
        help=(
            "Expected number of evaluation samples. Use 0 to disable "
            "this check. Default: 1002."
        ),
    )
    return parser.parse_args()


def detect_column(
    columns: Iterable[str],
    explicit_name: str | None,
    candidates: tuple[str, ...],
    role: str,
) -> str:
    available = list(columns)

    if explicit_name is not None:
        if explicit_name not in available:
            raise ValueError(
                f"{role} column '{explicit_name}' was not found. "
                f"Available columns: {available}"
            )
        return explicit_name

    lowered = {str(col).strip().lower(): col for col in available}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    raise ValueError(
        f"Could not automatically detect the {role} column. "
        f"Available columns: {available}. "
        f"Pass it explicitly with --{'true-col' if role == 'reference' else 'pred-col'}."
    )


def normalise_labels(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )


def validate_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_labels: list[str],
    unknown_label: str,
    expected_n: int,
) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("Reference and prediction arrays have different lengths.")

    if expected_n > 0 and len(y_true) != expected_n:
        raise ValueError(
            f"Expected {expected_n} rows, but found {len(y_true)}."
        )

    if pd.isna(y_true).any():
        raise ValueError("Reference labels contain missing values.")

    if pd.isna(y_pred).any():
        raise ValueError("Predictions contain missing values.")

    true_values = set(np.unique(y_true))
    invalid_true = true_values.difference(valid_labels)
    if invalid_true:
        raise ValueError(
            "Reference labels contain values outside the valid label set: "
            f"{sorted(invalid_true)}"
        )

    allowed_predictions = set(valid_labels) | {unknown_label}
    pred_values = set(np.unique(y_pred))
    invalid_pred = pred_values.difference(allowed_predictions)
    if invalid_pred:
        raise ValueError(
            "Predictions contain unsupported labels: "
            f"{sorted(invalid_pred)}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_labels: list[str],
) -> tuple[float, float]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=valid_labels,
        average="macro",
        zero_division=0,
    )
    return float(accuracy), float(macro_f1)


def percentile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lower = np.quantile(values, alpha / 2.0)
    upper = np.quantile(values, 1.0 - alpha / 2.0)
    return float(lower), float(upper)


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_labels: list[str],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    if n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be greater than zero.")

    if not 0.0 < confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = len(y_true)

    accuracy_boot = np.empty(n_bootstrap, dtype=float)
    macro_f1_boot = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sampled_true = y_true[indices]
        sampled_pred = y_pred[indices]

        accuracy_boot[i], macro_f1_boot[i] = compute_metrics(
            sampled_true,
            sampled_pred,
            valid_labels,
        )

    point_accuracy, point_macro_f1 = compute_metrics(
        y_true,
        y_pred,
        valid_labels,
    )
    accuracy_low, accuracy_high = percentile_interval(
        accuracy_boot,
        confidence,
    )
    macro_f1_low, macro_f1_high = percentile_interval(
        macro_f1_boot,
        confidence,
    )

    return {
        "accuracy": point_accuracy,
        "accuracy_ci_low": accuracy_low,
        "accuracy_ci_high": accuracy_high,
        "macro_f1": point_macro_f1,
        "macro_f1_ci_low": macro_f1_low,
        "macro_f1_ci_high": macro_f1_high,
    }


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        data = pd.read_csv(args.input)

        true_col = detect_column(
            data.columns,
            args.true_col,
            TRUE_COLUMN_CANDIDATES,
            "reference",
        )
        pred_col = detect_column(
            data.columns,
            args.pred_col,
            PRED_COLUMN_CANDIDATES,
            "prediction",
        )

        valid_labels = [str(label).strip().lower() for label in args.labels]
        unknown_label = str(args.unknown_label).strip().lower()

        y_true = normalise_labels(data[true_col]).to_numpy(dtype=object)
        y_pred = normalise_labels(data[pred_col]).to_numpy(dtype=object)

        validate_data(
            y_true=y_true,
            y_pred=y_pred,
            valid_labels=valid_labels,
            unknown_label=unknown_label,
            expected_n=args.expected_n,
        )

        results = bootstrap_metrics(
            y_true=y_true,
            y_pred=y_pred,
            valid_labels=valid_labels,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )

        unknown_count = int(np.sum(y_pred == unknown_label))
        unknown_rate = unknown_count / len(y_pred)

        output_row = {
            "model": args.model,
            "setting": args.setting,
            "n_samples": len(y_true),
            "true_column": true_col,
            "prediction_column": pred_col,
            "n_bootstrap": args.n_bootstrap,
            "confidence_level": args.confidence,
            "seed": args.seed,
            "accuracy": results["accuracy"],
            "accuracy_ci_low": results["accuracy_ci_low"],
            "accuracy_ci_high": results["accuracy_ci_high"],
            "macro_f1": results["macro_f1"],
            "macro_f1_ci_low": results["macro_f1_ci_low"],
            "macro_f1_ci_high": results["macro_f1_ci_high"],
            "unknown_count": unknown_count,
            "unknown_rate": unknown_rate,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([output_row]).to_csv(args.output, index=False)

        print(f"Model: {args.model}")
        print(f"Setting: {args.setting}")
        print(f"Samples: {len(y_true)}")
        print(f"Reference column: {true_col}")
        print(f"Prediction column: {pred_col}")
        print(
            "Accuracy: "
            f"{results['accuracy']:.3f} "
            f"[{results['accuracy_ci_low']:.3f}, "
            f"{results['accuracy_ci_high']:.3f}]"
        )
        print(
            "Macro-F1: "
            f"{results['macro_f1']:.3f} "
            f"[{results['macro_f1_ci_low']:.3f}, "
            f"{results['macro_f1_ci_high']:.3f}]"
        )
        print(
            f"Unknown: {unknown_count}/{len(y_pred)} "
            f"({unknown_rate:.3f})"
        )
        print(f"Saved: {args.output}")

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
