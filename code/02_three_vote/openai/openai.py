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
# CONFIGURATION — THREE-VOTE SELF-CONSISTENCY
# =============================================================================
VOTES = 3
REQUESTED_TEMPERATURE = 0.3

ALLOWED_LABELS = ["joy", "anger", "sadness", "fear", "neutral"]
UNKNOWN_LABEL = "unknown"
STRICT_UNKNOWN_SENTINEL = "__unknown__"
STRICT_LABELS = ALLOWED_LABELS + [UNKNOWN_LABEL]

PROVIDER_NAME = "OpenAI"

# All OpenAI models reported in the manuscript.
MODELS: dict[str, str] = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "GPT-5": "gpt-5",
    "OpenAI-o3": "o3",
}

MODEL_RUNTIME_CONFIG: dict[str, dict[str, Any]] = {
    "GPT-4o": {
        "max_workers": 5,
        "request_spacing": 0.00,
        "max_output_tokens": 160,
        "reasoning_effort": None,
    },
    "GPT-4.1-mini": {
        "max_workers": 5,
        "request_spacing": 0.00,
        "max_output_tokens": 160,
        "reasoning_effort": None,
    },
    "GPT-5": {
        "max_workers": 4,
        "request_spacing": 0.00,
        "max_output_tokens": 300,
        "reasoning_effort": "low",
    },
    "OpenAI-o3": {
        "max_workers": 4,
        "request_spacing": 0.00,
        "max_output_tokens": 400,
        "reasoning_effort": "low",
    },
}

MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 0.8
MAX_BACKOFF_SECONDS = 30.0
JITTER_SECONDS = 0.25
SEED = 42

STORE_TEXT = True
STORE_RAW_VOTES = True
PRINT_FIRST_N = 10

# The experiment requests T=0.3. Some reasoning-model API versions may reject
# temperature. In that case only, the request is retried without temperature,
# and the fallback is recorded in the output.
ALLOW_TEMPERATURE_FALLBACK = True

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "__THREE_VOTE"

random.seed(SEED)
PRINT_LOCK = threading.Lock()


# =============================================================================
# PROTOCOL REASONS
# =============================================================================
VOTE_OK = "ok"
VOTE_API_ERROR = "api_error"
VOTE_EMPTY_OUTPUT = "empty_output"
VOTE_INCOMPLETE_MAX_TOKENS = "incomplete_max_output_tokens"
VOTE_INVALID_JSON = "invalid_json"
VOTE_MISSING_LABEL = "invalid_json_missing_label"
VOTE_INVALID_LABEL = "invalid_label"
VOTE_REGEX_LABEL = "regex_label"

FINAL_MAJORITY = "majority"
FINAL_ALL_FAILED = "all_failed"
FINAL_ALL_DIFFERENT_111 = "all_different_111"
FINAL_NO_MAJORITY_11U = "no_majority_11u"


# =============================================================================
# PROMPT AND STRUCTURED OUTPUT SCHEMA
# =============================================================================
INSTRUCTIONS = (
    "You are a strict Arabic emotion classification engine. "
    "Classify the Arabic sentence into exactly one label from: "
    "joy, anger, sadness, fear, neutral. "
    "Return only one valid JSON object containing exactly one key named label. "
    "Do not include explanations, markdown, or additional keys."
)

PROMPT_TEMPLATE = """
Choose exactly ONE label from:
joy, anger, sadness, fear, neutral

Rules:
- Return valid JSON only.
- Use exactly this schema: {{"label":"joy"}}
- Use lowercase English labels only.
- Choose the single best label.
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
# RATE LIMITER
# =============================================================================
class AdaptiveRateLimiter:
    def __init__(
        self,
        initial_rps: float = 0.8,
        min_rps: float = 0.2,
        max_rps: float = 6.0,
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
            self._rps = min(self._max_rps, self._rps + 0.05)

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
# PATHS AND DATA
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "EmotionsFile_GoldLabelCSV.csv"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "three_votes"
PRED_DIR = OUTPUT_ROOT / "predictions"
FIG_DIR = OUTPUT_ROOT / "figures"
METRIC_DIR = OUTPUT_ROOT / "metrics"

for directory in (PRED_DIR, FIG_DIR, METRIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def normalize_arabic(text: Any) -> str:
    value = str(text)
    value = re.sub("[إأٱآا]", "ا", value)
    value = re.sub("ى", "ي", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    required = {"text", "emotion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "emotion"]).copy().reset_index(drop=True)

    aliases = {
        "angry": "anger",
        "mad": "anger",
        "happy": "joy",
        "hapy": "joy",
        "sad": "sadness",
    }

    df["text"] = df["text"].map(normalize_arabic)
    df["emotion"] = (
        df["emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda value: aliases.get(value, value))
    )

    invalid = sorted(set(df["emotion"]) - set(ALLOWED_LABELS))
    if invalid:
        raise ValueError(
            f"Invalid gold labels: {invalid}. Allowed labels: {ALLOWED_LABELS}"
        )

    if "sample_id" not in df.columns:
        df.insert(
            0,
            "sample_id",
            [f"sample_{index + 1:04d}" for index in range(len(df))],
        )
    else:
        df["sample_id"] = df["sample_id"].astype(str)

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

    first = label.split(" ")[0] if label else ""
    if first in ALLOWED_LABELS:
        return first

    for allowed in ALLOWED_LABELS:
        if re.search(rf"\b{re.escape(allowed)}\b", label):
            return allowed

    return UNKNOWN_LABEL


def parse_label(raw_text: str) -> tuple[str, str]:
    raw = (raw_text or "").strip()
    if not raw:
        return UNKNOWN_LABEL, VOTE_EMPTY_OUTPUT

    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    candidates = [cleaned]
    match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))

    for blob in candidates:
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
                return UNKNOWN_LABEL, VOTE_MISSING_LABEL

            label = clean_label(obj[label_key])
            if label in ALLOWED_LABELS:
                return label, VOTE_OK
            return UNKNOWN_LABEL, VOTE_INVALID_LABEL

    explicit = re.search(
        r"\blabel\b\s*[:=]\s*[\"']?\s*"
        r"(joy|anger|sadness|fear|neutral)\s*[\"']?",
        cleaned.lower(),
    )
    if explicit:
        return explicit.group(1), VOTE_REGEX_LABEL

    hits = [
        label
        for label in ALLOWED_LABELS
        if re.search(rf"\b{re.escape(label)}\b", cleaned.lower())
    ]
    if len(hits) == 1:
        return hits[0], VOTE_REGEX_LABEL

    return UNKNOWN_LABEL, VOTE_INVALID_JSON


def extract_response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: list[str] = []
    output = getattr(response, "output", None)

    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue

            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())

                parsed = getattr(part, "parsed", None)
                if parsed is not None:
                    chunks.append(json.dumps(parsed, ensure_ascii=False))

    return "\n".join(chunks).strip()


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
    return (
        "temperature" in message
        and any(
            marker in message
            for marker in ("unsupported", "not supported", "does not support")
        )
    )


def is_transient(status: Optional[int]) -> bool:
    return status is None or status in {408, 409, 429, 500, 502, 503, 504, 529}


# =============================================================================
# OPENAI REQUEST
# =============================================================================
def structured_text_config() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "emotion_label",
            "schema": JSON_SCHEMA,
            "strict": True,
        }
    }


def build_request_kwargs(
    model_id: str,
    sentence: str,
    runtime_config: dict[str, Any],
    include_temperature: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "instructions": INSTRUCTIONS,
        "input": PROMPT_TEMPLATE.format(TEXT=sentence),
        "text": structured_text_config(),
        "max_output_tokens": int(runtime_config["max_output_tokens"]),
    }

    reasoning_effort = runtime_config.get("reasoning_effort")
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    if include_temperature:
        kwargs["temperature"] = REQUESTED_TEMPERATURE

    return kwargs


def call_single_vote(
    client: OpenAI,
    limiter: AdaptiveRateLimiter,
    model_name: str,
    model_id: str,
    runtime_config: dict[str, Any],
    sentence: str,
    sample_id: str,
    vote_number: int,
) -> dict[str, Any]:
    salt = f"{model_name}|{sample_id}|vote_{vote_number}"
    include_temperature = True
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()

        spacing = float(runtime_config.get("request_spacing", 0.0))
        if spacing > 0:
            time.sleep(spacing)

        try:
            kwargs = build_request_kwargs(
                model_id=model_id,
                sentence=sentence,
                runtime_config=runtime_config,
                include_temperature=include_temperature,
            )
            response = client.responses.create(**kwargs)

            status = str(getattr(response, "status", "completed") or "completed")
            incomplete_details = getattr(response, "incomplete_details", None)
            incomplete_reason = (
                getattr(incomplete_details, "reason", str(incomplete_details))
                if incomplete_details is not None
                else ""
            )

            if status == "incomplete":
                reason = (
                    VOTE_INCOMPLETE_MAX_TOKENS
                    if incomplete_reason == "max_output_tokens"
                    else f"incomplete:{incomplete_reason}"
                )
                return {
                    "prediction": UNKNOWN_LABEL,
                    "reason": reason,
                    "raw": "",
                    "attempts": attempt,
                    "temperature_sent": include_temperature,
                    "response_status": status,
                    "error_message": reason,
                }

            raw = extract_response_text(response)
            prediction, parse_reason = parse_label(raw)

            if prediction in ALLOWED_LABELS:
                limiter.note_success()
                return {
                    "prediction": prediction,
                    "reason": parse_reason,
                    "raw": raw[:1000],
                    "attempts": attempt,
                    "temperature_sent": include_temperature,
                    "response_status": status,
                    "error_message": "",
                }

            last_error = parse_reason

            if attempt < MAX_RETRIES:
                sleep_backoff(attempt, salt)
                continue

        except Exception as error:
            status_code = get_status_code(error)
            last_error = f"{type(error).__name__}: {str(error)[:500]}"

            if (
                include_temperature
                and ALLOW_TEMPERATURE_FALLBACK
                and is_temperature_parameter_error(error)
            ):
                include_temperature = False
                continue

            if status_code == 429:
                retry_after = get_retry_after(error)
                limiter.note_429(retry_after)
                if retry_after is not None:
                    time.sleep(min(60.0, retry_after))

            if attempt < MAX_RETRIES and is_transient(status_code):
                sleep_backoff(attempt, salt)
                continue

            return {
                "prediction": UNKNOWN_LABEL,
                "reason": VOTE_API_ERROR,
                "raw": "",
                "attempts": attempt,
                "temperature_sent": include_temperature,
                "response_status": str(status_code or "error"),
                "error_message": last_error,
            }

    return {
        "prediction": UNKNOWN_LABEL,
        "reason": last_error or VOTE_API_ERROR,
        "raw": "",
        "attempts": MAX_RETRIES,
        "temperature_sent": include_temperature,
        "response_status": "failed",
        "error_message": last_error,
    }


# =============================================================================
# VOTE AGGREGATION
# =============================================================================
def vote_pattern(counter: Counter) -> str:
    if not counter:
        return "0"
    return "-".join(map(str, sorted(counter.values(), reverse=True)))


def aggregate_votes(vote_predictions: list[str]) -> tuple[str, str, str, int]:
    valid_votes = [
        prediction
        for prediction in vote_predictions
        if prediction in ALLOWED_LABELS
    ]
    counts = Counter(valid_votes)
    unknown_votes = VOTES - len(valid_votes)
    pattern = vote_pattern(counts)

    if not valid_votes:
        return UNKNOWN_LABEL, FINAL_ALL_FAILED, pattern, unknown_votes

    if len(valid_votes) == VOTES and len(counts) == VOTES:
        return UNKNOWN_LABEL, FINAL_ALL_DIFFERENT_111, pattern, unknown_votes

    top_label, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        return top_label, FINAL_MAJORITY, pattern, unknown_votes

    return UNKNOWN_LABEL, FINAL_NO_MAJORITY_11U, pattern, unknown_votes


def classify_sample(
    client: OpenAI,
    limiter: AdaptiveRateLimiter,
    row: pd.Series,
    model_name: str,
    model_id: str,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    vote_results = [
        call_single_vote(
            client=client,
            limiter=limiter,
            model_name=model_name,
            model_id=model_id,
            runtime_config=runtime_config,
            sentence=row["text"],
            sample_id=row["sample_id"],
            vote_number=vote_number,
        )
        for vote_number in range(1, VOTES + 1)
    ]

    vote_predictions = [result["prediction"] for result in vote_results]
    final_prediction, final_reason, pattern, unknown_votes = aggregate_votes(
        vote_predictions
    )

    return {
        "sample_id": row["sample_id"],
        "text": row["text"],
        "gold_label": row["emotion"],
        "prediction": final_prediction,
        "is_correct": int(final_prediction == row["emotion"]),
        "abstained": int(final_prediction == UNKNOWN_LABEL),
        "vote_pattern": pattern,
        "unknown_votes": unknown_votes,
        "unknown_reason": final_reason,
        "calls_made": sum(int(result["attempts"]) for result in vote_results),
        "provider": PROVIDER_NAME,
        "model": model_name,
        "model_id": model_id,
        "run_id": RUN_ID,
        "votes": VOTES,
        "temp_votes": REQUESTED_TEMPERATURE,
        "max_workers": int(runtime_config["max_workers"]),
        "votes_list": json.dumps(vote_predictions, ensure_ascii=False),
        "raw_votes": json.dumps(
            [result["raw"] for result in vote_results],
            ensure_ascii=False,
        ),
        "vote_reasons": json.dumps(
            [result["reason"] for result in vote_results],
            ensure_ascii=False,
        ),
        "temperature_sent_votes": json.dumps(
            [result["temperature_sent"] for result in vote_results]
        ),
        "response_statuses": json.dumps(
            [result["response_status"] for result in vote_results]
        ),
        "error_messages": json.dumps(
            [result["error_message"] for result in vote_results],
            ensure_ascii=False,
        ),
    }


# =============================================================================
# METRICS AND FIGURES
# =============================================================================
def compute_metrics(predictions_df: pd.DataFrame) -> dict[str, Any]:
    gold = predictions_df["gold_label"].astype(str).tolist()
    predictions = predictions_df["prediction"].astype(str).tolist()

    strict_predictions = [
        prediction
        if prediction in ALLOWED_LABELS
        else STRICT_UNKNOWN_SENTINEL
        for prediction in predictions
    ]

    strict_accuracy = accuracy_score(gold, strict_predictions)
    strict_macro_f1 = f1_score(
        gold,
        strict_predictions,
        labels=ALLOWED_LABELS,
        average="macro",
        zero_division=0,
    )
    strict_micro_f1 = f1_score(
        gold,
        strict_predictions,
        labels=ALLOWED_LABELS,
        average="micro",
        zero_division=0,
    )
    strict_weighted_f1 = f1_score(
        gold,
        strict_predictions,
        labels=ALLOWED_LABELS,
        average="weighted",
        zero_division=0,
    )

    covered = predictions_df[
        predictions_df["prediction"].isin(ALLOWED_LABELS)
    ].copy()

    if not covered.empty:
        covered_accuracy = accuracy_score(
            covered["gold_label"],
            covered["prediction"],
        )
        covered_macro_f1 = f1_score(
            covered["gold_label"],
            covered["prediction"],
            labels=ALLOWED_LABELS,
            average="macro",
            zero_division=0,
        )
        covered_micro_f1 = f1_score(
            covered["gold_label"],
            covered["prediction"],
            labels=ALLOWED_LABELS,
            average="micro",
            zero_division=0,
        )
        covered_weighted_f1 = f1_score(
            covered["gold_label"],
            covered["prediction"],
            labels=ALLOWED_LABELS,
            average="weighted",
            zero_division=0,
        )
    else:
        covered_accuracy = 0.0
        covered_macro_f1 = 0.0
        covered_micro_f1 = 0.0
        covered_weighted_f1 = 0.0

    total = len(predictions_df)
    unknown_total = int(
        (predictions_df["prediction"] == UNKNOWN_LABEL).sum()
    )

    return {
        "total_samples": total,
        "strict_accuracy_unknown_wrong": strict_accuracy,
        "strict_macro_f1": strict_macro_f1,
        "strict_micro_f1": strict_micro_f1,
        "strict_weighted_f1": strict_weighted_f1,
        "coverage": 1.0 - (unknown_total / total),
        "unknown_rate": unknown_total / total,
        "unknown_total": unknown_total,
        "covered_accuracy": covered_accuracy,
        "covered_macro_f1": covered_macro_f1,
        "covered_micro_f1": covered_micro_f1,
        "covered_weighted_f1": covered_weighted_f1,
    }


def save_outputs(
    predictions_df: pd.DataFrame,
    model_name: str,
    metrics: dict[str, Any],
) -> None:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")

    prediction_path = (
        PRED_DIR
        / f"{PROVIDER_NAME}__RUN{RUN_ID}__{safe_model}__PREDICTIONS.csv"
    )
    predictions_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = classification_report(
        predictions_df["gold_label"],
        [
            prediction
            if prediction in ALLOWED_LABELS
            else STRICT_UNKNOWN_SENTINEL
            for prediction in predictions_df["prediction"]
        ],
        labels=ALLOWED_LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        METRIC_DIR
        / f"{PROVIDER_NAME}__RUN{RUN_ID}__{safe_model}__REPORT_STRICT.csv",
        encoding="utf-8-sig",
    )

    reason_distribution = (
        predictions_df["unknown_reason"]
        .value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="count")
    )
    reason_distribution.to_csv(
        METRIC_DIR
        / f"{PROVIDER_NAME}__RUN{RUN_ID}__{safe_model}__UNKNOWN_REASONS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(
        predictions_df["gold_label"],
        predictions_df["prediction"],
        labels=STRICT_LABELS,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(STRICT_LABELS)))
    ax.set_yticks(range(len(STRICT_LABELS)))
    ax.set_xticklabels(STRICT_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(STRICT_LABELS)

    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            ax.text(
                column_index,
                row_index,
                str(cm[row_index, column_index]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        FIG_DIR
        / f"{PROVIDER_NAME}__RUN{RUN_ID}__{safe_model}__CM_STRICT_WITH_UNKNOWN.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    metrics_path = (
        METRIC_DIR
        / f"{PROVIDER_NAME}__RUN{RUN_ID}__{safe_model}__METRICS.json"
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Put it in a local .env file "
            "or environment variable."
        )

    df = load_dataset()
    client = OpenAI(api_key=api_key)

    distribution = (
        df["emotion"]
        .value_counts()
        .reindex(ALLOWED_LABELS, fill_value=0)
        .rename_axis("label")
        .reset_index(name="count")
    )
    distribution.to_csv(
        METRIC_DIR / f"{PROVIDER_NAME}__RUN{RUN_ID}__DATASET_DISTRIBUTION.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows: list[dict[str, Any]] = []

    for model_name, model_id in MODELS.items():
        runtime_config = MODEL_RUNTIME_CONFIG[model_name]
        limiter = AdaptiveRateLimiter()
        records: list[Optional[dict[str, Any]]] = [None] * len(df)

        print(
            f"\nRunning {model_name}: votes={VOTES}, "
            f"requested_temperature={REQUESTED_TEMPERATURE}, "
            f"workers={runtime_config['max_workers']}"
        )

        # Small compatibility test before the full run.
        test_result = call_single_vote(
            client=client,
            limiter=limiter,
            model_name=model_name,
            model_id=model_id,
            runtime_config=runtime_config,
            sentence=normalize_arabic("انا مبسوطة اليوم"),
            sample_id="TEST",
            vote_number=1,
        )
        print(
            f"[TEST] prediction={test_result['prediction']} | "
            f"reason={test_result['reason']} | "
            f"temperature_sent={test_result['temperature_sent']}"
        )
        if test_result["prediction"] == UNKNOWN_LABEL:
            raise RuntimeError(
                f"Test call failed for {model_name}: "
                f"{test_result['error_message'] or test_result['reason']}"
            )

        def task(index: int) -> tuple[int, dict[str, Any]]:
            record = classify_sample(
                client=client,
                limiter=limiter,
                row=df.iloc[index],
                model_name=model_name,
                model_id=model_id,
                runtime_config=runtime_config,
            )
            return index, record

        with ThreadPoolExecutor(
            max_workers=int(runtime_config["max_workers"])
        ) as executor:
            futures = {
                executor.submit(task, index): index
                for index in range(len(df))
            }

            for completed_number, future in enumerate(
                tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=model_name,
                ),
                start=1,
            ):
                index = futures[future]

                try:
                    result_index, record = future.result()
                    records[result_index] = record
                except Exception as error:
                    row = df.iloc[index]
                    records[index] = {
                        "sample_id": row["sample_id"],
                        "text": row["text"],
                        "gold_label": row["emotion"],
                        "prediction": UNKNOWN_LABEL,
                        "is_correct": 0,
                        "abstained": 1,
                        "vote_pattern": "0",
                        "unknown_votes": VOTES,
                        "unknown_reason": FINAL_ALL_FAILED,
                        "calls_made": 0,
                        "provider": PROVIDER_NAME,
                        "model": model_name,
                        "model_id": model_id,
                        "run_id": RUN_ID,
                        "votes": VOTES,
                        "temp_votes": REQUESTED_TEMPERATURE,
                        "max_workers": int(runtime_config["max_workers"]),
                        "votes_list": json.dumps(
                            [UNKNOWN_LABEL] * VOTES
                        ),
                        "raw_votes": json.dumps([""] * VOTES),
                        "vote_reasons": json.dumps(
                            ["future_exception"] * VOTES
                        ),
                        "temperature_sent_votes": json.dumps(
                            [False] * VOTES
                        ),
                        "response_statuses": json.dumps(
                            ["future_exception"] * VOTES
                        ),
                        "error_messages": json.dumps(
                            [
                                f"{type(error).__name__}: "
                                f"{str(error)[:500]}"
                            ]
                            * VOTES
                        ),
                    }

                if completed_number <= PRINT_FIRST_N:
                    current = records[index]
                    with PRINT_LOCK:
                        print(
                            f"[{model_name}] {current['sample_id']} | "
                            f"gold={current['gold_label']} | "
                            f"pred={current['prediction']} | "
                            f"reason={current['unknown_reason']}"
                        )

        if any(record is None for record in records):
            raise RuntimeError(
                f"Internal error: missing records for {model_name}."
            )

        predictions_df = pd.DataFrame(records)
        if not STORE_TEXT:
            predictions_df = predictions_df.drop(columns=["text"])

        metrics = compute_metrics(predictions_df)
        reason_counter = Counter(predictions_df["unknown_reason"])

        metrics.update(
            {
                "provider": PROVIDER_NAME,
                "model_name": model_name,
                "model_id": model_id,
                "run_id": RUN_ID,
                "setup": (
                    "OpenAI Responses API + structured JSON + "
                    "three-vote self-consistency (requested T=0.3) + "
                    "valid-vote aggregation"
                ),
                "votes": VOTES,
                "temp_votes_requested": REQUESTED_TEMPERATURE,
                "max_workers": int(runtime_config["max_workers"]),
                "reasoning_effort": runtime_config["reasoning_effort"],
                "unknown_reason_all_failed": int(
                    reason_counter.get(FINAL_ALL_FAILED, 0)
                ),
                "unknown_reason_all_different_111": int(
                    reason_counter.get(FINAL_ALL_DIFFERENT_111, 0)
                ),
                "unknown_reason_no_majority_11u": int(
                    reason_counter.get(FINAL_NO_MAJORITY_11U, 0)
                ),
                "unknown_reason_majority": int(
                    reason_counter.get(FINAL_MAJORITY, 0)
                ),
                "rows_all_votes_with_temperature": int(
                    predictions_df["temperature_sent_votes"]
                    .map(lambda value: all(json.loads(value)))
                    .sum()
                ),
            }
        )

        save_outputs(
            predictions_df=predictions_df,
            model_name=model_name,
            metrics=metrics,
        )
        summary_rows.append(metrics)

        print(f"Finished {model_name}")
        print(
            f"Strict accuracy={metrics['strict_accuracy_unknown_wrong']:.4f}"
        )
        print(f"Strict macro-F1={metrics['strict_macro_f1']:.4f}")
        print(f"Unknown rate={metrics['unknown_rate']:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = (
        METRIC_DIR
        / f"{PROVIDER_NAME}__RUN{RUN_ID}__SUMMARY_THREE_VOTE.csv"
    )
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "run_id": RUN_ID,
        "provider": PROVIDER_NAME,
        "models": MODELS,
        "model_runtime_config": MODEL_RUNTIME_CONFIG,
        "votes": VOTES,
        "requested_temperature": REQUESTED_TEMPERATURE,
        "temperature_fallback_enabled": ALLOW_TEMPERATURE_FALLBACK,
        "allowed_labels": ALLOWED_LABELS,
        "dataset_path": str(DATA_PATH),
        "n_samples": len(df),
    }
    (
        METRIC_DIR
        / f"{PROVIDER_NAME}__RUN{RUN_ID}__CONFIG_THREE_VOTE.json"
    ).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nOpenAI three-vote benchmark completed successfully.")
    print("RUN_ID:", RUN_ID)
    print("Models run:", list(MODELS.keys()))
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()