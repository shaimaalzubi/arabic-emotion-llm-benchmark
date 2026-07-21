# Deployment-Oriented Evaluation of Large Language Models for Arabic Emotion Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research Software](https://img.shields.io/badge/type-research%20software-blue.svg)](CITATION.cff)
[![Zenodo DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#citation)

## Overview

This repository contains the code and documentation associated with the study:

**Deployment-Oriented Evaluation of Large Language Models for Arabic Emotion Classification**

The benchmark evaluates Arabic emotion classification across five labels:

```text
anger
joy
sadness
fear
neutral
```

The repository documents the following experimental components:

1. zero-shot single-pass inference (`k=1`, temperature zero where explicitly supported);
2. few-shot single-pass inference using five external demonstrations (`k=1`, temperature zero where explicitly supported);
3. three-vote self-consistency using three independent generations (`T=0.3` where supported);
4. GPT-4o run-to-run stability analysis across three complete zero-shot runs at temperature zero;
5. supervised AraBERT and MARBERT reference baselines;
6. exact pairwise McNemar testing with Holm correction.

The public release does **not** include:

- the full Arabic text dataset;
- original sample-level prediction outputs;
- original provider logs;
- original generated figures;
- API credentials.

## Evaluated Models

### OpenAI API

- GPT-4o
- GPT-4.1-mini
- GPT-5
- OpenAI-o3

### Anthropic API

- Claude-Sonnet-4.6

### Google Gemini API

- Gemini-2.5-Flash
- Gemma-3-12B

### Groq API

- Qwen-3-32B
- Allam-2-7B
- LLaMA-3.1-8B

### Mistral API

- Mistral-Medium
- Mistral-Large

The study was not designed as a tier-matched provider comparison. Results should be interpreted as applying to the specific model and API configurations evaluated, not as provider-level rankings.

## Dataset

The benchmark contains **1,002 manually curated Arabic sentences**.

The final reference label is stored in the `emotion` column after adjudication.

| Emotion   |   Samples |  Percentage |
| --------- | --------: | ----------: |
| Anger     |       251 |      25.05% |
| Neutral   |       237 |      23.65% |
| Joy       |       207 |      20.66% |
| Sadness   |       169 |      16.87% |
| Fear      |       138 |      13.77% |
| **Total** | **1,002** | **100.00%** |

Two annotators independently labelled all samples before adjudication:

- agreements: `980`;
- disagreements: `22`;
- observed agreement: `0.9780`;
- expected agreement: `0.2094`;
- Cohen's kappa: `0.9722`.

The full text dataset is **not included in this public repository** because the sentences were derived from publicly accessible social-media content and may still raise privacy, platform-policy, ethical, and redistribution considerations.

Authorised access may be requested from the corresponding author, subject to applicable institutional and ethical requirements.

See:

```text
docs/DATASET.md
docs/COHEN_KAPPA.md
```

## Repository Structure

```text
BnchmarkZenodo/
├── .env.example
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
│
├── code/
│   ├── 01_one_vote/
│   │   ├── anthropic/
│   │   ├── google/
│   │   ├── groq/
│   │   ├── mistral/
│   │   └── openai/
│   │
│   ├── 02_three_vote/
│   │   ├── anthropic/
│   │   ├── google/
│   │   ├── groq/
│   │   ├── mistral/
│   │   └── openai/
│   │
│   ├── 03_few_shot_single_inference/
│   │   ├── anthropic/
│   │   ├── google/
│   │   ├── groq/
│   │   ├── mistral/
│   │   └── openai/
│   │
│   ├── 04_gpt_rerun/
│   │   └── gptRerun.py
│   │
│   ├── 05_AraBert_MarBert/
│   │   ├── AraBERT.ipynb
│   │   ├── MarBERT.ipynb
│   │   ├── ARABERT_README.md
│   │   └── MARBERT_README.md
│   │
│   └── McNemar/
│       └── mcNemar.py
│
├── data/
│
├── docs/
│   ├── ACKNOWLEDGEMENTS.md
│   ├── CODE_OF_CONDUCT.md
│   ├── COHEN_KAPPA.md
│   ├── CONTRIBUTING.md
│   ├── DATASET.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── PROJECT_STRUCTURE.md
│   └── REPRODUCIBILITY.md
│
└── output/
    ├── few_shot/
    ├── GPT_runs/
    ├── one_vote/
    └── three_votes/
```

For a detailed description of every directory and file, see:

```text
docs/PROJECT_STRUCTURE.md
```

## Installation

Clone the repository and create a Python environment.

```bash
git clone https://github.com/shaimaalzubi/Benchmark.git
cd Benchmark

python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, call the environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe <SCRIPT_PATH>
```

### Windows Command Prompt

```bat
.venv\Scripts\activate.bat
```

### macOS or Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Hosted-provider SDK behaviour may depend on package versions, account access, and provider-side API changes.

## API Configuration

Copy:

```text
.env.example
```

to:

```text
.env
```

and add only the credentials required for the scripts you intend to run:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
MISTRAL_API_KEY=
```

Never commit or upload:

- `.env`;
- API keys;
- credentials;
- request headers;
- service-account files;
- private provider logs.

## Dataset Placement for Authorised Reproduction

After obtaining authorised access, place the dataset at:

```text
data/EmotionsFile_GoldLabelCSV.csv
```

The required columns are:

```text
text
emotion
```

The `emotion` column is the final adjudicated reference label.

Before execution, verify:

- exactly 1,002 valid rows;
- no missing required values;
- UTF-8-compatible encoding;
- only the five valid reference labels;
- class counts matching `docs/DATASET.md`.

## Experimental Protocols

### Zero-shot single-pass evaluation

Location:

```text
code/01_one_vote/
```

Protocol:

- one inference request per sample;
- `k=1`;
- zero-shot prompt;
- temperature zero where explicitly supported;
- provider-specific validation and retry handling.

Recommended output location:

```text
output/one_vote/
```

### Few-shot single-pass evaluation

Location:

```text
code/03_few_shot_single_inference/
```

Protocol:

- five manually constructed demonstrations;
- one example per emotion class;
- demonstrations external to the evaluation dataset;
- one inference request per sample;
- `k=1`;
- temperature zero where explicitly supported.

Recommended output location:

```text
output/few_shot/
```

### Three-vote self-consistency evaluation

Location:

```text
code/02_three_vote/
```

Protocol:

- three independent generations per sample;
- sampling temperature `T=0.3` where supported;
- validation of each generated response;
- valid-vote aggregation;
- a final valid emotion only when at least two votes agree.

Final harmonised unknown outcomes:

```text
all_failed
all_different_111
no_majority_11u
```

Recommended output location:

```text
output/three_votes/
```

### GPT-4o run-to-run stability

Location:

```text
code/04_gpt_rerun/gptRerun.py
```

Protocol:

- GPT-4o;
- zero-shot prompt;
- three complete runs;
- temperature zero;
- comparison of consistency, variability, persistent errors, and correctness changes.

Recommended output location:

```text
output/GPT_runs/
```

### Supervised AraBERT and MARBERT baselines

Location:

```text
code/05_AraBert_MarBert/
```

Reported split:

```text
training:   801
validation: 100
test:       101
```

Before public release, notebook outputs must be cleared if they display restricted dataset text, local credentials, or local absolute paths.

### McNemar significance analysis

Location:

```text
code/McNemar/mcNemar.py
```

Reported analysis:

- 12 evaluated models;
- same 1,002 zero-shot samples;
- 66 pairwise comparisons;
- two-sided exact McNemar test;
- Holm step-down correction;
- adjusted threshold `p<0.05`.

The original prediction files required to reproduce the exact manuscript table are not included in this public repository.

## Running Scripts

Run commands from the repository root only after verifying each script's:

- dataset path;
- output path;
- provider model identifier;
- API access;
- parameter support;
- quota;
- retry configuration.

Example script locations:

```text
code/01_one_vote/openai/gpt.py
code/02_three_vote/google/gemma.py
code/03_few_shot_single_inference/mistral/mistral.py
code/04_gpt_rerun/gptRerun.py
code/McNemar/mcNemar.py
```

The repository documentation uses the root-level paths:

```text
data/
output/
```

Any script that still contains a local or condition-specific path must be updated or configured before execution.

Detailed execution guidance is provided in:

```text
docs/REPRODUCIBILITY.md
```

## Output Policy

The `output/` directory is reserved for files generated by new executions.

Original sample-level prediction outputs, derived evaluation files, provider logs, and generated figures from the reported study are **not included** in this public repository.

Recommended generated-output structure:

```text
output/<EXPERIMENT>/predictions/
output/<EXPERIMENT>/metrics/
output/<EXPERIMENT>/logs/
output/<EXPERIMENT>/figures/
```

New executions should use clearly identified run folders and should be described as:

- re-runs;
- replications;
- extensions;
- corrections.

They must not be described as the original archived outputs.

## Evaluation Summary

### Strict evaluation

All 1,002 samples remain in the evaluation.

A final `unknown` prediction is treated as incorrect.

### Covered evaluation

Samples with final `unknown` predictions are excluded before predictive metrics are calculated.

### Coverage

```text
valid predictions / total samples
```

### Unknown rate

```text
unknown predictions / total samples
```

For this benchmark:

```text
coverage = 1 - unknown rate
```

### Three-vote aggregation

A valid emotion label is selected only when it receives at least two of the three votes.

The label `unknown` is an evaluation outcome and is not one of the five gold emotion classes.

## Reproducibility Limitations

Exact hosted-model reproduction may be affected by:

- silent model revisions;
- model alias changes;
- backend serving changes;
- provider-side nondeterminism;
- parameter-support differences;
- rate limits;
- quota restrictions;
- API deprecation;
- safety filtering;
- regional availability;
- model-access changes.

A new run should preserve:

- exact model identifier;
- provider;
- access date;
- requested parameters;
- parameters actually sent;
- prompt version;
- parser version;
- retry policy;
- run identifier;
- dataset checksum;
- output checksum.

## Documentation

- [`docs/DATASET.md`](docs/DATASET.md)  
  Dataset size, labels, class distribution, agreement, access, privacy, and integrity checks.

- [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md)  
  Benchmark logic, strict and covered evaluation, unknown handling, and vote aggregation.

- [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)  
  Repository tree and file-level descriptions.

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)  
  Environment setup, execution procedures, logging, output policy, and replication guidance.

- [`docs/COHEN_KAPPA.md`](docs/COHEN_KAPPA.md)  
  Inter-annotator agreement calculation.

- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)  
  Contribution and generated-output rules.

- [`docs/ACKNOWLEDGEMENTS.md`](docs/ACKNOWLEDGEMENTS.md)  
  Contributors, institutions, and software providers.

## Citation

The Zenodo DOI will be inserted after the final repository release is archived.

```text
Al-Shamaileh, O., Alzubi, S., Almehrzi, M., Mubin, O., & Alnajjar, F. (2026).
Deployment-Oriented Evaluation of Large Language Models for Arabic Emotion Classification.
Software, version 1.0.0. Zenodo. DOI: pending.
```

Machine-readable citation metadata are available in:

```text
CITATION.cff
```

After DOI assignment, update:

- `CITATION.cff`;
- this README;
- the manuscript's Code Availability section;
- the final Zenodo record.

## License

The source code is released under the [MIT License](LICENSE).

The MIT License applies to the software only.

It does not automatically grant permission to redistribute:

- the underlying social-media text;
- restricted dataset material;
- third-party content;
- provider-generated outputs;
- content governed by external platform terms.
