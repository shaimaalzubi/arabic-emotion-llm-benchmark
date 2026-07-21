# Reproducibility Guide

## Overview

This repository supports the study:

**Deployment-Oriented Evaluation of Large Language Models for Arabic Emotion Classification**

The public release contains code and documentation for re-running the evaluation pipeline. It does **not** include:

- the full Arabic text dataset;
- original sample-level prediction outputs;
- original provider logs;
- original generated figures;
- API credentials.

Because hosted model APIs may change over time, a new execution should be described as a re-run or replication rather than as an exact recreation of the original hosted-model outputs.

## Experimental Components

The repository documents six experimental components:

1. zero-shot single-pass evaluation;
2. few-shot single-pass evaluation;
3. three-vote self-consistency evaluation;
4. GPT-4o run-to-run stability analysis;
5. supervised AraBERT and MARBERT baselines;
6. pairwise exact McNemar significance testing.

## Repository Requirements

Before execution, verify that the repository contains:

```text
code/
data/
docs/
output/
README.md
requirements.txt
.env.example
```

The expected public structure is documented in:

```text
docs/PROJECT_STRUCTURE.md
```

## Environment Setup

Create and activate a Python environment from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell script execution is restricted, the environment's interpreter can be called directly:

```powershell
.\.venv\Scripts\python.exe <SCRIPT_PATH>
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Provider Credentials

Copy:

```text
.env.example
```

to:

```text
.env
```

and add only the credentials required for the provider scripts that will be executed.

Typical environment variables may include:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY
GROQ_API_KEY
MISTRAL_API_KEY
```

Do not commit or upload the `.env` file.

Do not place API keys directly inside Python scripts, notebooks, documentation, screenshots, logs, or generated outputs.

## Dataset Access and Placement

The full text dataset is not distributed in the public repository.

After authorised access is obtained, place the dataset at:

```text
data/EmotionsFile_GoldLabelCSV.csv
```

The required columns are:

```text
text
emotion
```

The `emotion` column contains the final adjudicated reference label.

The valid reference labels are:

```text
anger
joy
sadness
fear
neutral
```

The evaluation label `unknown` is not a gold emotion class.

Detailed dataset documentation is provided in:

```text
docs/DATASET.md
```

## Dataset Integrity Checks

Before running any experiment, verify:

- total valid rows: `1002`;
- no missing `text` values;
- no missing `emotion` values;
- all labels belong to the five-class set;
- the final class counts match the documented distribution;
- the file is read using a compatible UTF-8 encoding.

Expected class counts:

```text
anger:   251
neutral: 237
joy:     207
sadness: 169
fear:    138
```

Example verification:

```python
import pandas as pd

df = pd.read_csv(
    "data/EmotionsFile_GoldLabelCSV.csv",
    encoding="utf-8-sig"
)

assert len(df) == 1002
assert {"text", "emotion"}.issubset(df.columns)
assert df["text"].notna().all()
assert df["emotion"].notna().all()

labels = (
    df["emotion"]
    .astype(str)
    .str.strip()
    .str.lower()
)

valid_labels = {"anger", "joy", "sadness", "fear", "neutral"}
assert set(labels) <= valid_labels

expected_counts = {
    "anger": 251,
    "neutral": 237,
    "joy": 207,
    "sadness": 169,
    "fear": 138,
}

assert labels.value_counts().to_dict() == expected_counts
```

## Model and API Verification

Before executing a hosted-model script, confirm:

- the model identifier is still available;
- the provider account has access to the model;
- the API key has sufficient permissions;
- the account has sufficient quota or credit;
- the provider supports the requested parameters;
- the script records any parameter fallback;
- the configured output directory exists or can be created.

Hosted-model names and aliases may change. Any substitute model must be documented explicitly and must not be presented as the original evaluated configuration.

## Common Output Schema

Where supported by the provider, models are instructed to return:

```json
{ "label": "<emotion>" }
```

Valid labels are restricted to:

```text
anger
joy
sadness
fear
neutral
```

An output may be classified as `unknown` when it cannot be mapped to a valid label because of:

- malformed or incomplete structured output;
- a missing label field;
- an unsupported label;
- an empty response;
- a blocked response;
- an API or HTTP failure;
- a parsing failure;
- output truncation;
- exhausted retries.

Provider-specific parsers and structured-output mechanisms may differ. Any such difference must be preserved in the execution record and considered when interpreting unknown rates.

## Experiment 1: Zero-Shot Single-Pass Evaluation

### Location

```text
code/01_one_vote/
```

### Protocol

- one inference request per sample;
- `k=1`;
- zero-shot prompting;
- five valid emotion labels;
- temperature set to zero where explicitly supported by the provider interface;
- provider-specific retry and validation procedures;
- strict and covered evaluation.

### Provider Scripts

```text
code/01_one_vote/anthropic/claude.py
code/01_one_vote/google/gemma.py
code/01_one_vote/groq/groq_file.py
code/01_one_vote/mistral/mistral.py
code/01_one_vote/openai/gpt.py
```

### Recommended Output Location

```text
output/one_vote/
```

Recommended subdirectories:

```text
output/one_vote/predictions/
output/one_vote/metrics/
output/one_vote/logs/
output/one_vote/figures/
```

### Required Records

For each sample, preserve where available:

```text
sample_id
gold_label
prediction
model
provider
run_id
raw_response_status
unknown_reason
temperature_requested
temperature_sent
timestamp
```

Do not include restricted dataset text in the public output unless redistribution has been explicitly authorised.

## Experiment 2: Few-Shot Single-Pass Evaluation

### Location

```text
code/03_few_shot_single_inference/
```

### Protocol

- five manually constructed demonstrations;
- one demonstration for each emotion class;
- demonstrations external to the 1,002-sample evaluation dataset;
- one inference request per sample;
- `k=1`;
- temperature set to zero where explicitly supported;
- strict evaluation used for the primary zero-shot versus few-shot comparison.

### Provider Scripts

```text
code/03_few_shot_single_inference/anthropic/claude.py
code/03_few_shot_single_inference/google/gemini.py
code/03_few_shot_single_inference/groq/groq_file.py
code/03_few_shot_single_inference/mistral/mistral.py
code/03_few_shot_single_inference/openai/gpt.py
```

### Recommended Output Location

```text
output/few_shot/
```

Recommended subdirectories:

```text
output/few_shot/predictions/
output/few_shot/metrics/
output/few_shot/logs/
output/few_shot/figures/
```

### Prompt Verification

Before execution, verify:

- the same five emotion classes are represented;
- no demonstration appears in the evaluation dataset;
- the expected JSON output format is documented;
- any provider-specific wording difference is retained;
- any provider-level parameter restriction is logged.

## Experiment 3: Three-Vote Self-Consistency Evaluation

### Location

```text
code/02_three_vote/
```

### Protocol

- three independent generations per sample;
- sampling temperature `T=0.3` where supported;
- each generated response validated separately;
- intermediate invalid responses mapped to `unknown`;
- valid-vote aggregation;
- a final valid emotion retained only when at least two votes agree.

### Provider Scripts

```text
code/02_three_vote/anthropic/claude.py
code/02_three_vote/google/gemma.py
code/02_three_vote/groq/groq_file.py
code/02_three_vote/mistral/mistral.py
code/02_three_vote/openai/gpt.py
```

### Recommended Output Location

```text
output/three_votes/
```

Recommended subdirectories:

```text
output/three_votes/predictions/
output/three_votes/metrics/
output/three_votes/logs/
output/three_votes/figures/
```

### Final Aggregation Outcomes

The harmonised final unknown categories are:

```text
all_failed
all_different_111
no_majority_11u
```

Definitions:

- `all_failed`  
  all three responses were non-evaluable;

- `all_different_111`  
  three different valid emotion labels were produced;

- `no_majority_11u`  
  the available valid votes did not form a two-vote majority because one or more votes were non-evaluable.

When at least two valid votes agree, record:

```text
majority
```

### Required Vote-Level Records

Preserve where available:

```text
sample_id
gold_label
vote_1
vote_2
vote_3
final_prediction
final_reason
model
provider
run_id
temperature_requested
temperature_sent_vote_1
temperature_sent_vote_2
temperature_sent_vote_3
technical_failure_vote_1
technical_failure_vote_2
technical_failure_vote_3
```

This distinction is important because the final unknown category and the underlying provider-level technical reason are not conceptually identical.

## Experiment 4: GPT-4o Run-to-Run Stability

### Location

```text
code/04_gpt_rerun/
```

### Main Script

```text
code/04_gpt_rerun/gptRerun.py
```

### Protocol

- GPT-4o;
- zero-shot prompt;
- full 1,002-sample dataset;
- three complete runs;
- one response per sample per run;
- temperature zero;
- aligned prompt and parsing configuration.

### Reported Stability Measures

The associated study reports:

- mean accuracy and standard deviation;
- mean macro-F1 and standard deviation;
- consistency;
- variability;
- persistent error rate;
- correctness-change rate.

### Recommended Output Location

```text
output/GPT_runs/
```

Recommended subdirectories:

```text
output/GPT_runs/predictions/
output/GPT_runs/metrics/
output/GPT_runs/logs/
```

A repeated run at temperature zero may still differ because of provider-side nondeterminism, backend changes, distributed inference, or silent model updates.

## Experiment 5: Supervised Arabic Encoder Baselines

### Location

```text
code/05_AraBert_MarBert/
```

### Files

```text
code/05_AraBert_MarBert/AraBERT.ipynb
code/05_AraBert_MarBert/ARABERT_README.md
code/05_AraBert_MarBert/MarBERT.ipynb
code/05_AraBert_MarBert/MARBERT_README.md
```

### Reported Data Split

```text
training:   801
validation: 100
test:       101
```

The split was stratified by the five reference emotion classes.

### Reported AraBERT Configuration

- four epochs;
- AdamW;
- learning rate `2e-5`;
- weight decay `0.01`;
- training batch size `16`;
- evaluation batch size `16`;
- checkpoint evaluation after each epoch;
- best model selected using validation macro-F1.

### Reported MARBERT Configuration

- one epoch;
- AdamW;
- learning rate `2e-5`;
- weight decay `0.01`;
- training batch size `4`;
- evaluation batch size `4`;
- no early stopping;
- final trained model used for evaluation.

### Public Notebook Requirements

Before public release:

- clear all executed outputs that display dataset text;
- remove cached credentials;
- remove local absolute paths;
- restore all required initialization cells;
- verify execution from a clean runtime;
- ensure that any model checkpoint path is documented;
- avoid including restricted dataset rows in screenshots or cell outputs.

## Experiment 6: Pairwise McNemar Analysis

### Location

```text
code/McNemar/mcNemar.py
```

### Protocol

- paired zero-shot predictions;
- same 1,002 samples for every model pair;
- 12 evaluated models;
- 66 pairwise comparisons;
- binary correctness outcome per sample;
- two-sided exact McNemar test;
- Holm step-down correction;
- adjusted significance threshold `p<0.05`.

### Required Prediction Schema

Each input prediction file must contain:

```text
sample_id or dataset_index
gold_label
prediction
```

The same sample identifier and row alignment must be used across all model files.

### Recommended Input Location

```text
output/one_vote/predictions/
```

### Recommended Analysis Output

```text
output/mcnemar_12_models/
```

Recommended files:

```text
pairwise_mcnemar_full.csv
pairwise_mcnemar_selected.csv
holm_adjusted_pvalues.csv
mcnemar_heatmap.png
analysis_log.txt
```

The public repository does not include the original prediction files required to recreate the manuscript's exact McNemar table.

## Strict Evaluation

Under strict evaluation:

- all 1,002 samples remain in the evaluation;
- valid emotion predictions are scored normally;
- `unknown` predictions are treated as incorrect.

## Covered Evaluation

Under covered evaluation:

- samples with final prediction `unknown` are excluded;
- predictive metrics are calculated only on samples with valid emotion predictions.

Coverage is:

```text
number of valid predictions / total number of samples
```

Unknown rate is:

```text
number of unknown predictions / total number of samples
```

For this benchmark:

```text
coverage = 1 - unknown rate
```

## Performance Metrics

The evaluation scripts may calculate:

- accuracy;
- macro-F1;
- weighted F1;
- coverage;
- unknown rate;
- strict accuracy;
- strict macro-F1;
- covered accuracy;
- covered macro-F1.

The manuscript emphasises macro-F1 because the benchmark is multiclass and moderately imbalanced.

## Output Directory Policy

The public repository reserves these locations:

```text
output/one_vote/
output/few_shot/
output/three_votes/
output/GPT_runs/
```

The original outputs from the reported study are not included.

New executions should use clearly named run folders, for example:

```text
output/one_vote/predictions/run_2026-07-21/
output/three_votes/logs/run_2026-07-21/
```

Do not describe newly generated files as the original archived results.

Do not overwrite prior generated results without documenting:

- execution date;
- code version;
- model identifier;
- provider;
- prompt version;
- requested parameters;
- sent parameters;
- dataset checksum;
- output checksum.

## Recommended Run Manifest

Each execution should include a manifest such as:

```json
{
  "experiment": "three_vote",
  "run_id": "run_2026-07-21",
  "dataset_rows": 1002,
  "dataset_sha256": "<SHA256>",
  "model": "<MODEL_ID>",
  "provider": "<PROVIDER>",
  "temperature_requested": 0.3,
  "temperature_sent": 0.3,
  "votes_per_sample": 3,
  "prompt_version": "<VERSION>",
  "code_commit": "<COMMIT_HASH>",
  "started_at": "<ISO_TIMESTAMP>",
  "completed_at": "<ISO_TIMESTAMP>"
}
```

Where a provider does not accept an explicit parameter, record:

```json
{
  "temperature_requested": 0.3,
  "temperature_sent": null,
  "parameter_note": "Explicit temperature parameter not accepted by provider interface."
}
```

## Recommended Validation Before a Full Run

Before processing all 1,002 samples:

1. run a small subset;
2. verify dataset loading;
3. verify API authentication;
4. verify the model identifier;
5. inspect raw responses;
6. verify parser behaviour;
7. verify output paths;
8. verify unknown-reason logging;
9. verify that retries do not duplicate completed rows;
10. confirm that resumed runs preserve original sample indices.

A small test run must not be reported as part of the final benchmark.

## Error Handling

Provider scripts may implement:

- retries;
- exponential backoff;
- rate-limit handling;
- timeout recovery;
- empty-response detection;
- structured-output recovery;
- unsupported-parameter fallback;
- partial-run resumption.

Any fallback that changes the submitted request must be logged.

A provider error should not be silently converted into a valid emotion prediction.

## Provider-Specific Reproducibility Limits

Exact reproduction of hosted-model outputs may be impossible because of:

- silent model revisions;
- model alias changes;
- infrastructure changes;
- stochastic serving behaviour;
- unsupported parameter changes;
- API deprecation;
- safety filtering;
- quota restrictions;
- rate limits;
- model availability;
- regional access differences;
- backend nondeterminism.

Therefore, report:

- exact model identifier used;
- provider;
- date of access;
- requested temperature;
- temperature actually sent;
- retry policy;
- parser version;
- prompt version;
- run identifier.

## Result Comparison

The aggregate values in the associated manuscript represent the original experimental results.

A new execution should be compared with those aggregate values, but it must not be assumed to reproduce:

- every sample-level prediction;
- every unknown output;
- every vote pattern;
- every provider failure;
- every McNemar discordance count.

Differences must be reported transparently.

## Privacy and Redistribution

Do not publish:

- unauthorised dataset text;
- usernames;
- account identifiers;
- post URLs;
- traceable timestamps;
- API keys;
- private provider logs;
- notebook outputs exposing restricted text.

The repository licence applies to software only.

## Reproducibility Checklist

Before publishing or depositing a release, verify:

- [ ] The dataset is not included without authorisation.
- [ ] Dataset documentation reports 1,002 samples.
- [ ] Class counts are 251, 237, 207, 169, and 138.
- [ ] Cohen's kappa documentation reports `0.9722`.
- [ ] Zero-shot is documented as single-pass with temperature zero where supported.
- [ ] Few-shot is documented as five demonstrations and single-pass.
- [ ] Three-vote is documented as three generations at `T=0.3` where supported.
- [ ] GPT-4o stability analysis remains documented at temperature zero.
- [ ] Unknown is not described as a gold emotion class.
- [ ] Strict and covered evaluation are defined.
- [ ] Original sample-level outputs are not claimed to be included.
- [ ] Notebook outputs do not expose restricted text.
- [ ] API credentials are absent.
- [ ] Output directories are described as reserved for new runs.
- [ ] Model identifiers and access dates are documented.
- [ ] Parameter fallback is logged.
- [ ] McNemar input requirements are documented.
- [ ] The Zenodo DOI is added after deposition.
- [ ] `CITATION.cff` matches the final release.

## Citation

When using this software or documentation, cite the software record using:

```text
CITATION.cff
```

After Zenodo deposition, add the assigned DOI to:

- `CITATION.cff`;
- `README.md`;
- the manuscript's Code Availability section;
- any release notes.
