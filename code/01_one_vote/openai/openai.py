from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================
VOTES = 1
REQUESTED_TEMPERATURE = 0.0

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_UNKNOWN_SENTINEL = "__unknown__"

PROVIDER_NAME = "OpenAI"

# Canonical model names used in the manuscript.
MODELS: dict[str, str] = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "GPT-5": "gpt-5",
    "OpenAI-o3": "o3",
}

# Reasoning models may reject the temperature argument on some API versions.
# The code first requests temperature=0.0; if the API explicitly rejects that
# parameter, it retries the same request without temperature and records this.
ALLOW_TEMPERATURE_FALLBACK = True

MAX_WORKERS = 3
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 0.8
MAX_BACKOFF_SECONDS = 30.0
JITTER_SECONDS = 0.40
REQUEST_SPACING_SECONDS = 0.0

INITIAL_RPS = 0.8
MIN_RPS = 0.2
MAX_RPS = 6.0

MAX_OUTPUT_TOKENS = 512
SEED = 42

# Keep False for a public repository if the dataset text is not public.
INCLUDE_TEXT_IN_OUTPUT = False

PRINT_FIRST_N = 10
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__ZERO_SHOT_V1"

random.seed(SEED)
PRINT_LOCK = threading.Lock()


# =============================================================================
# PROMPT AND JSON SCHEMA
# =============================================================================
INSTRUCTIONS = (
    "You are an Arabic emotion classification engine. "
    "Classify each Arabic sentence into exactly one of these labels: "
    "joy, anger, sadness, fear, neutral. "
    "Return only a valid JSON object containing exactly one key named label. "
    "Do not include explanations, markdown, or additional keys."
)

PROMPT_TEMPLATE = """
Choose exactly ONE label from:
joy, anger, sadness, fear, neutral

Rules:
- Return valid JSON only.
- Use exactly this schema: {{"label":"joy"}}
- Use lowercase English labels only.
- If uncertain, choose the single best label from the five allowed labels.
- Never output unknown.

Arabic sentence:
{TEXT}
""".strip()

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ALLOWED_LABELS,
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}


# =============================================================================
# RESULT OBJECT
# =============================================================================
@dataclass
class PredictionResult:
    prediction: str
    unknown_reason: str
    attempts: int
    temperature_sent: bool
    response_status: str
    error_message: str


# =============================================================================
# ADAPTIVE GLOBAL RATE LIMITER
# =============================================================================
class AdaptiveRateLimiter:
    def __init__(
        self,
        initial_rps: float = INITIAL_RPS,
        min_rps: float = MIN_RPS,
        max_rps: float = MAX_RPS,
    ) -> None:
        self._lock = threading.Lock()
        self._rps = float(initial_rps)
        self._min_rps = float(min_rps)
        self._max_rps = float(max_rps)
        self._cooldown_until = 0.0
        self._last_request_time = 0.0
        self._consecutive_429 = 0

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.time()

                if now < self._cooldown_until:
                    sleep_for = self._cooldown_until - now
                else:
                    interval = 1.0 / max(self._min_rps, self._rps)
                    elapsed = now - self._last_request_time

                    if self._last_request_time == 0.0 or elapsed >= interval:
                        self._last_request_time = now
                        return

                    sleep_for = interval - elapsed

            time.sleep(max(0.0, sleep_for))

    def note_success(self) -> None:
        with self._lock:
            self._consecutive_429 = max(0, self._consecutive_429 - 1)
            self._rps = min(self._max_rps, self._rps + 0.10)

    def note_429(self, retry_after: Optional[float]) -> None:
        with self._lock:
            self._consecutive_429 += 1
            reduction = 0.75 if self._consecutive_429 == 1 else 0.60
            self._rps = max(self._min_rps, self._rps * reduction)

            cooldown = (
                float(retry_after)
                if retry_after is not None
                else min(20.0, 1.5 * self._consecutive_429)
            )
            self._cooldown_until = max(
                self._cooldown_until,
                time.time() + max(0.0, cooldown),
            )


# =============================================================================
# PATHS AND ARGUMENTS
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI zero-shot Arabic emotion classification, one vote."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the CSV dataset. It must contain text and emotion columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <repository_root>/output/one_vote",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Display names of the OpenAI models to run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of concurrent sample requests. Default: {MAX_WORKERS}",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include original text in prediction CSVs. Avoid this for public release.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows for testing. Omit for all samples.",
    )
    return parser.parse_args()


def resolve_data_path(user_path: Optional[str]) -> Path:
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    filename = "EmotionsFile_GoldLabelCSV.csv"

    candidates = [
        project_root / "data" / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find EmotionsFile_GoldLabelCSV.csv. Searched:\n"
        f"{searched}\nUse --data to provide its path."
    )


def prepare_output_dirs(user_output_dir: Optional[str]) -> dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    root = (
        Path(user_output_dir).expanduser().resolve()
        if user_output_dir
        else project_root / "output" / "one_vote"
    )

    dirs = {
        "root": root,
        "predictions": root / "predictions",
        "metrics": root / "metrics",
        "figures": root / "figures",
        "debug": root / "debug",
    }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================
def normalize_arabic(text: Any) -> str:
    value = str(text)
    value = re.sub("[إأٱآا]", "ا", value)
    value = re.sub("ى", "ي", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def load_dataset(data_path: Path, limit: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(data_path, encoding="utf-8")

    required = {"text", "emotion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "emotion"]).copy().reset_index(drop=True)

    label_aliases = {
        "angry": "anger",
        "mad": "anger",
        "happy": "joy",
        "hapy": "joy",
        "sad": "sadness",
    }

    df["emotion"] = (
        df["emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: label_aliases.get(x, x))
    )

    invalid_gold = sorted(set(df["emotion"]) - set(ALLOWED_LABELS))
    if invalid_gold:
        raise ValueError(
            f"Invalid gold labels: {invalid_gold}. Allowed labels: {ALLOWED_LABELS}"
        )

    df["text"] = df["text"].map(normalize_arabic)

    # Stable public-safe row identifier. It does not expose the sentence.
    if "sample_id" not in df.columns:
        df.insert(0, "sample_id", [f"sample_{i + 1:04d}" for i in range(len(df))])
    else:
        df["sample_id"] = df["sample_id"].astype(str)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero.")
        df = df.head(limit).copy().reset_index(drop=True)

    return df


# =============================================================================
# RESPONSE PARSING
# =============================================================================
def clean_label(value: Any) -> str:
    if value is None:
        return UNKNOWN_LABEL

    label = str(value).strip().lower()
    label = re.sub(r"[^a-z_-]", " ", label)
    label = re.sub(r"\s+", " ", label).strip()

    if label in ALLOWED_LABELS:
        return label

    first_token = label.split(" ")[0] if label else ""
    if first_token in ALLOWED_LABELS:
        return first_token

    for allowed in ALLOWED_LABELS:
        if re.search(rf"\b{re.escape(allowed)}\b", label):
            return allowed

    return UNKNOWN_LABEL


def parse_label(raw_text: str) -> tuple[str, str]:
    raw = (raw_text or "").strip()
    if not raw:
        return UNKNOWN_LABEL, "empty_response"

    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "").strip()

    objects_to_try = [raw]
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if match and match.group(0) != raw:
        objects_to_try.append(match.group(0))

    for blob in objects_to_try:
        for candidate in (blob, blob.replace("'", '"')):
            try:
                obj = json.loads(candidate)
            except Exception:
                continue

            if not isinstance(obj, dict):
                continue

            label_key = next(
                (key for key in obj if str(key).strip().lower() == "label"),
                None,
            )
            if label_key is None:
                continue

            label = clean_label(obj[label_key])
            if label in ALLOWED_LABELS:
                return label, "ok"
            return UNKNOWN_LABEL, "invalid_label"

    return UNKNOWN_LABEL, "invalid_json"


def extract_response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)

    return ""


# =============================================================================
# RETRY HELPERS
# =============================================================================
def deterministic_jitter(attempt: int, salt: str) -> float:
    digest = hashlib.md5(f"{SEED}|{attempt}|{salt}".encode("utf-8")).hexdigest()
    fraction = int(digest[:8], 16) / float(16**8)
    return fraction * JITTER_SECONDS


def sleep_backoff(attempt: int, salt: str) -> None:
    delay = min(
        MAX_BACKOFF_SECONDS,
        BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)),
    )
    time.sleep(delay + deterministic_jitter(attempt, salt))


def get_status_code(error: Exception) -> Optional[int]:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def get_retry_after(error: Exception) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    for key in ("retry-after", "Retry-After"):
        value = headers.get(key)
        if value is not None:
            try:
                return float(str(value).strip().replace("s", ""))
            except ValueError:
                pass
    return None


def is_temperature_parameter_error(error: Exception) -> bool:
    message = str(error).lower()
    markers = (
        "temperature",
        "unsupported parameter",
        "does not support",
        "not supported",
    )
    return "temperature" in message and any(marker in message for marker in markers[1:])


def is_transient(status: Optional[int]) -> bool:
    return status is None or status in {408, 409, 429, 500, 502, 503, 504, 529}


# =============================================================================
# OPENAI CALL — EXACTLY ONE GENERATION PER SAMPLE
# =============================================================================
def text_format_config() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "emotion_label",
            "schema": JSON_SCHEMA,
            "strict": True,
        }
    }


def reasoning_config(model_id: str) -> Optional[dict[str, str]]:
    # Reasoning configuration is used only for GPT-5 and o3.
    if model_id == "gpt-5" or model_id.startswith("o3"):
        return {"effort": "low"}
    return None


def build_request_kwargs(
    model_id: str,
    sentence: str,
    include_temperature: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "instructions": INSTRUCTIONS,
        "input": PROMPT_TEMPLATE.format(TEXT=sentence),
        "text": text_format_config(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }

    reasoning = reasoning_config(model_id)
    if reasoning is not None:
        kwargs["reasoning"] = reasoning

    if include_temperature:
        kwargs["temperature"] = REQUESTED_TEMPERATURE

    return kwargs


def classify_one(
    client: OpenAI,
    limiter: AdaptiveRateLimiter,
    model_name: str,
    model_id: str,
    sentence: str,
    sample_id: str,
) -> PredictionResult:
    """Perform one successful generation attempt for the sample.

    Retries are transport/format recovery only. They are not experimental votes.
    The final row always contains one prediction.
    """

    salt = f"{model_name}|{sample_id}"
    last_error = ""
    include_temperature = True

    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        if REQUEST_SPACING_SECONDS > 0:
            time.sleep(REQUEST_SPACING_SECONDS)

        try:
            kwargs = build_request_kwargs(
                model_id=model_id,
                sentence=sentence,
                include_temperature=include_temperature,
            )
            response = client.responses.create(**kwargs)

            status = str(getattr(response, "status", "completed") or "completed")
            raw_text = extract_response_text(response)
            prediction, parse_reason = parse_label(raw_text)

            if prediction in ALLOWED_LABELS:
                limiter.note_success()
                return PredictionResult(
                    prediction=prediction,
                    unknown_reason="ok",
                    attempts=attempt,
                    temperature_sent=include_temperature,
                    response_status=status,
                    error_message="",
                )

            last_error = parse_reason
            # Invalid/empty formatting can be retried as API recovery.
            sleep_backoff(attempt, salt)

        except Exception as error:
            status_code = get_status_code(error)
            last_error = f"{type(error).__name__}: {str(error)[:300]}"

            if (
                include_temperature
                and ALLOW_TEMPERATURE_FALLBACK
                and is_temperature_parameter_error(error)
            ):
                include_temperature = False
                # Immediate compatibility retry; still one experimental prediction.
                continue

            if status_code == 429:
                retry_after = get_retry_after(error)
                limiter.note_429(retry_after)
                if retry_after:
                    time.sleep(min(60.0, retry_after))

            if is_transient(status_code):
                sleep_backoff(attempt, salt)
                continue

            return PredictionResult(
                prediction=UNKNOWN_LABEL,
                unknown_reason="api_error",
                attempts=attempt,
                temperature_sent=include_temperature,
                response_status=str(status_code or "error"),
                error_message=last_error,
            )

    return PredictionResult(
        prediction=UNKNOWN_LABEL,
        unknown_reason="max_retries_exceeded",
        attempts=MAX_RETRIES,
        temperature_sent=include_temperature,
        response_status="failed",
        error_message=last_error,
    )


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(gold: pd.Series, predictions: pd.Series) -> dict[str, Any]:
    gold_values = gold.astype(str).tolist()
    pred_values = predictions.astype(str).tolist()

    strict_predictions = [
        prediction if prediction in ALLOWED_LABELS else STRICT_UNKNOWN_SENTINEL
        for prediction in pred_values
    ]

    strict_accuracy = accuracy_score(gold_values, strict_predictions)
    strict_macro_f1 = f1_score(
        gold_values,
        strict_predictions,
        labels=ALLOWED_LABELS,
        average="macro",
        zero_division=0,
    )
    strict_weighted_f1 = f1_score(
        gold_values,
        strict_predictions,
        labels=ALLOWED_LABELS,
        average="weighted",
        zero_division=0,
    )

    covered_mask = [prediction in ALLOWED_LABELS for prediction in pred_values]
    covered_gold = [g for g, keep in zip(gold_values, covered_mask) if keep]
    covered_pred = [p for p, keep in zip(pred_values, covered_mask) if keep]

    if covered_gold:
        covered_accuracy = accuracy_score(covered_gold, covered_pred)
        covered_macro_f1 = f1_score(
            covered_gold,
            covered_pred,
            labels=ALLOWED_LABELS,
            average="macro",
            zero_division=0,
        )
        covered_weighted_f1 = f1_score(
            covered_gold,
            covered_pred,
            labels=ALLOWED_LABELS,
            average="weighted",
            zero_division=0,
        )
    else:
        covered_accuracy = float("nan")
        covered_macro_f1 = float("nan")
        covered_weighted_f1 = float("nan")

    total = len(gold_values)
    covered_count = len(covered_gold)
    unknown_count = total - covered_count

    return {
        "n_samples": total,
        "covered_count": covered_count,
        "unknown_count": unknown_count,
        "coverage": covered_count / total if total else float("nan"),
        "unknown_rate": unknown_count / total if total else float("nan"),
        "strict_accuracy": strict_accuracy,
        "strict_macro_f1": strict_macro_f1,
        "strict_weighted_f1": strict_weighted_f1,
        "covered_accuracy": covered_accuracy,
        "covered_macro_f1": covered_macro_f1,
        "covered_weighted_f1": covered_weighted_f1,
    }


def save_confusion_matrix(
    gold: pd.Series,
    predictions: pd.Series,
    model_name: str,
    output_path: Path,
) -> None:
    # Unknown predictions are not shown as a sixth class in this paper matrix.
    covered = predictions.isin(ALLOWED_LABELS)
    cm = confusion_matrix(
        gold[covered],
        predictions[covered],
        labels=ALLOWED_LABELS,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set_title(f"{model_name} — Zero-shot One-vote")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_xticks(range(len(ALLOWED_LABELS)))
    ax.set_yticks(range(len(ALLOWED_LABELS)))
    ax.set_xticklabels(ALLOWED_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(ALLOWED_LABELS)

    threshold = cm.max() / 2.0 if cm.size and cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MODEL RUNNER
# =============================================================================
def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def run_model(
    client: OpenAI,
    df: pd.DataFrame,
    model_name: str,
    model_id: str,
    workers: int,
    include_text: bool,
    output_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    limiter = AdaptiveRateLimiter()
    records: list[Optional[dict[str, Any]]] = [None] * len(df)

    def task(row_index: int) -> tuple[int, dict[str, Any]]:
        row = df.iloc[row_index]
        result = classify_one(
            client=client,
            limiter=limiter,
            model_name=model_name,
            model_id=model_id,
            sentence=row["text"],
            sample_id=row["sample_id"],
        )

        record: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "gold_label": row["emotion"],
            "prediction": result.prediction,
            "is_correct": int(result.prediction == row["emotion"]),
            "abstained": int(result.prediction == UNKNOWN_LABEL),
            "unknown_reason": result.unknown_reason,
            "calls_made": result.attempts,
            "provider": PROVIDER_NAME,
            "model": model_name,
            "model_id": model_id,
            "run_id": RUN_ID,
            "votes": VOTES,
            "temperature_requested": REQUESTED_TEMPERATURE,
            "temperature_sent": result.temperature_sent,
            "max_workers": workers,
            "response_status": result.response_status,
            "error_message": result.error_message,
        }

        if include_text:
            record["text"] = row["text"]

        return row_index, record

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(task, index): index
            for index in range(len(df))
        }

        for completed_number, future in enumerate(
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{model_name}",
            ),
            start=1,
        ):
            row_index = futures[future]
            try:
                result_index, record = future.result()
                records[result_index] = record
            except Exception as error:
                row = df.iloc[row_index]
                records[row_index] = {
                    "sample_id": row["sample_id"],
                    "gold_label": row["emotion"],
                    "prediction": UNKNOWN_LABEL,
                    "is_correct": 0,
                    "abstained": 1,
                    "unknown_reason": "future_exception",
                    "calls_made": 0,
                    "provider": PROVIDER_NAME,
                    "model": model_name,
                    "model_id": model_id,
                    "run_id": RUN_ID,
                    "votes": VOTES,
                    "temperature_requested": REQUESTED_TEMPERATURE,
                    "temperature_sent": False,
                    "max_workers": workers,
                    "response_status": "future_exception",
                    "error_message": f"{type(error).__name__}: {str(error)[:300]}",
                    **({"text": row["text"]} if include_text else {}),
                }

            if completed_number <= PRINT_FIRST_N:
                current = records[row_index]
                with PRINT_LOCK:
                    print(
                        f"[{model_name}] {current['sample_id']} | "
                        f"gold={current['gold_label']} | "
                        f"pred={current['prediction']} | "
                        f"reason={current['unknown_reason']}"
                    )

    if any(record is None for record in records):
        raise RuntimeError(f"Internal error: missing records for {model_name}.")

    predictions_df = pd.DataFrame(records)
    metrics = compute_metrics(
        predictions_df["gold_label"],
        predictions_df["prediction"],
    )

    metrics.update(
        {
            "provider": PROVIDER_NAME,
            "model": model_name,
            "model_id": model_id,
            "run_id": RUN_ID,
            "votes": VOTES,
            "temperature_requested": REQUESTED_TEMPERATURE,
            "rows_with_temperature_sent": int(
                predictions_df["temperature_sent"].sum()
            ),
            "rows_without_temperature_sent": int(
                (~predictions_df["temperature_sent"]).sum()
            ),
        }
    )

    model_file = safe_filename(model_name)
    prediction_path = (
        output_dirs["predictions"]
        / f"OpenAI__RUN{RUN_ID}__{model_file}__PREDICTIONS.csv"
    )
    metrics_path = (
        output_dirs["metrics"]
        / f"OpenAI__RUN{RUN_ID}__{model_file}__METRICS.json"
    )
    report_path = (
        output_dirs["metrics"]
        / f"OpenAI__RUN{RUN_ID}__{model_file}__CLASSIFICATION_REPORT.txt"
    )
    figure_path = (
        output_dirs["figures"]
        / f"OpenAI__RUN{RUN_ID}__{model_file}__CONFUSION_MATRIX.png"
    )

    predictions_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    covered = predictions_df["prediction"].isin(ALLOWED_LABELS)
    report = classification_report(
        predictions_df.loc[covered, "gold_label"],
        predictions_df.loc[covered, "prediction"],
        labels=ALLOWED_LABELS,
        zero_division=0,
    )
    report_path.write_text(report, encoding="utf-8")

    save_confusion_matrix(
        gold=predictions_df["gold_label"],
        predictions=predictions_df["prediction"],
        model_name=model_name,
        output_path=figure_path,
    )

    print(f"\nFinished {model_name}")
    print(f"Strict accuracy:  {metrics['strict_accuracy']:.4f}")
    print(f"Strict macro-F1:  {metrics['strict_macro_f1']:.4f}")
    print(f"Covered accuracy: {metrics['covered_accuracy']:.4f}")
    print(f"Covered macro-F1: {metrics['covered_macro_f1']:.4f}")
    print(f"Unknown rate:     {metrics['unknown_rate']:.4f}")
    print(f"Predictions:      {prediction_path}\n")

    return predictions_df, metrics


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Put it in a local .env file or environment variable."
        )

    data_path = resolve_data_path(args.data)
    output_dirs = prepare_output_dirs(args.output_dir)
    include_text = bool(args.include_text or INCLUDE_TEXT_IN_OUTPUT)

    print("=" * 78)
    print("OPENAI ZERO-SHOT ARABIC EMOTION CLASSIFICATION — ONE VOTE")
    print("=" * 78)
    print(f"Run ID:                 {RUN_ID}")
    print(f"Dataset:                {data_path}")
    print(f"Output root:            {output_dirs['root']}")
    print(f"Models:                 {args.models}")
    print(f"Votes per sample:       {VOTES}")
    print(f"Requested temperature: {REQUESTED_TEMPERATURE}")
    print(f"Include text in output: {include_text}")
    print("=" * 78)

    df = load_dataset(data_path, args.limit)
    print(f"Loaded samples: {len(df)}")
    print("Gold distribution:")
    print(df["emotion"].value_counts().reindex(ALLOWED_LABELS, fill_value=0))

    distribution_path = output_dirs["metrics"] / f"RUN{RUN_ID}__DATASET_DISTRIBUTION.csv"
    (
        df["emotion"]
        .value_counts()
        .reindex(ALLOWED_LABELS, fill_value=0)
        .rename_axis("label")
        .reset_index(name="count")
        .to_csv(distribution_path, index=False, encoding="utf-8-sig")
    )

    client = OpenAI(api_key=api_key)
    summary_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        model_id = MODELS[model_name]
        _, metrics = run_model(
            client=client,
            df=df,
            model_name=model_name,
            model_id=model_id,
            workers=max(1, args.workers),
            include_text=include_text,
            output_dirs=output_dirs,
        )
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    preferred_columns = [
        "provider",
        "model",
        "model_id",
        "n_samples",
        "strict_accuracy",
        "strict_macro_f1",
        "strict_weighted_f1",
        "covered_accuracy",
        "covered_macro_f1",
        "covered_weighted_f1",
        "coverage",
        "unknown_rate",
        "unknown_count",
        "votes",
        "temperature_requested",
        "rows_with_temperature_sent",
        "rows_without_temperature_sent",
        "run_id",
    ]
    summary_df = summary_df[
        [column for column in preferred_columns if column in summary_df.columns]
    ]

    summary_path = output_dirs["metrics"] / f"OpenAI__RUN{RUN_ID}__SUMMARY.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    run_config = {
        "run_id": RUN_ID,
        "provider": PROVIDER_NAME,
        "models": {name: MODELS[name] for name in args.models},
        "votes": VOTES,
        "requested_temperature": REQUESTED_TEMPERATURE,
        "temperature_fallback_enabled": ALLOW_TEMPERATURE_FALLBACK,
        "max_workers": max(1, args.workers),
        "max_retries": MAX_RETRIES,
        "dataset_path": str(data_path),
        "n_samples": len(df),
        "include_text_in_output": include_text,
        "allowed_labels": ALLOWED_LABELS,
        "gold_column": "emotion",
        "text_column": "text",
    }
    config_path = output_dirs["metrics"] / f"OpenAI__RUN{RUN_ID}__CONFIG.json"
    config_path.write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print("ALL REQUESTED MODELS COMPLETED")
    print(f"Summary: {summary_path}")
    print(f"Config:  {config_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
