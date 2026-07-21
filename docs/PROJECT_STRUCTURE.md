# Project Structure

## Overview

This repository accompanies the study:

**Deployment-Oriented Evaluation of Large Language Models for Arabic Emotion Classification**

The repository is organised around six experimental components:

1. zero-shot single-pass evaluation;
2. three-vote self-consistency evaluation;
3. few-shot single-pass evaluation;
4. GPT-4o run-to-run stability analysis;
5. supervised AraBERT and MARBERT baselines;
6. pairwise McNemar significance testing.

The repository contains code and documentation. The authorised dataset and the original sample-level outputs from the reported experiments are not included in the public release.

## Repository Tree

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
│   │   │   └── claude.py
│   │   ├── google/
│   │   │   └── gemma.py
│   │   ├── groq/
│   │   │   └── groq_file.py
│   │   ├── mistral/
│   │   │   └── mistral.py
│   │   └── openai/
│   │       └── gpt.py
│   │
│   ├── 02_three_vote/
│   │   ├── anthropic/
│   │   │   └── claude.py
│   │   ├── google/
│   │   │   └── gemma.py
│   │   ├── groq/
│   │   │   └── groq_file.py
│   │   ├── mistral/
│   │   │   └── mistral.py
│   │   └── openai/
│   │       └── gpt.py
│   │
│   ├── 03_few_shot_single_inference/
│   │   ├── anthropic/
│   │   │   └── claude.py
│   │   ├── google/
│   │   │   └── gemini.py
│   │   ├── groq/
│   │   │   └── groq_file.py
│   │   ├── mistral/
│   │   │   └── mistral.py
│   │   └── openai/
│   │       └── gpt.py
│   │
│   ├── 04_gpt_rerun/
│   │   ├── code/
│   │   └── gptRerun.py
│   │
│   ├── 05_AraBert_MarBert/
│   │   ├── AraBERT.ipynb
│   │   ├── ARABERT_README.md
│   │   ├── MarBERT.ipynb
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

## Root Files

### `.env.example`

Template for locally configured provider credentials.

The real `.env` file must not be committed or uploaded. API keys and secrets must remain local.

### `.gitignore`

Defines files and directories that should not be committed, including local credentials, temporary files, generated caches, and environment-specific artefacts.

### `CITATION.cff`

Machine-readable citation metadata for the software repository.

The Zenodo DOI and final publication metadata should be updated after deposition and publication.

### `LICENSE`

Software licence for the repository code.

The licence does not automatically apply to the underlying social-media text, third-party content, hosted-model outputs, or material governed by provider terms.

### `README.md`

Primary repository documentation, including:

- project scope;
- environment setup;
- dataset placement;
- supported experiments;
- execution guidance;
- limitations;
- citation instructions.

### `requirements.txt`

Python package requirements used by the scripts and notebooks.

Exact reproduction may additionally require provider SDK versions and model access equivalent to those available during the original experiments.

## `code/01_one_vote/`

Contains the zero-shot, single-pass evaluation scripts.

The reported primary benchmark used:

- one inference request per sample;
- `k=1`;
- a five-label emotion space;
- temperature set to zero where the provider interface accepted an explicit temperature parameter;
- provider-specific output validation and retry handling.

### Provider folders

- `anthropic/claude.py`  
  Anthropic API evaluation.

- `google/gemma.py`  
  Google API evaluation for the relevant Google-hosted configuration.

- `groq/groq_file.py`  
  Groq-hosted open-weight model evaluation.

- `mistral/mistral.py`  
  Mistral API evaluation.

- `openai/gpt.py`  
  OpenAI API evaluation.

## `code/02_three_vote/`

Contains the three-vote self-consistency scripts.

The reported protocol used:

- three independent generations per sample;
- sampling temperature `T=0.3` where supported;
- validation of every generated response;
- valid-vote aggregation;
- a final valid emotion only when at least two votes agreed.

The harmonised final unknown outcomes are:

```text
all_failed
all_different_111
no_majority_11u
```

Provider-specific technical failure details may also be retained in newly generated logs.

### Provider folders

- `anthropic/claude.py`
- `google/gemma.py`
- `groq/groq_file.py`
- `mistral/mistral.py`
- `openai/gpt.py`

## `code/03_few_shot_single_inference/`

Contains the few-shot, single-pass evaluation scripts.

The reported setup used:

- five manually constructed demonstrations;
- one example per emotion category;
- demonstrations external to the 1,002-sample evaluation dataset;
- one inference request per sample;
- `k=1`;
- temperature set to zero where explicitly supported.

### Provider folders

- `anthropic/claude.py`
- `google/gemini.py`
- `groq/groq_file.py`
- `mistral/mistral.py`
- `openai/gpt.py`

## `code/04_gpt_rerun/`

Contains the GPT-4o run-to-run stability analysis.

### `gptRerun.py`

Used to repeat the zero-shot GPT-4o evaluation under aligned settings and compare:

- accuracy across runs;
- macro-F1 across runs;
- prediction consistency;
- prediction variability;
- persistent errors;
- correctness changes.

The associated analysis used three complete zero-shot runs at temperature zero.

### `code/`

Currently an empty reserved subdirectory. It is not required by the documented public workflow unless future supporting files are added.

## `code/05_AraBert_MarBert/`

Contains the supervised Arabic encoder baseline notebooks and model-specific documentation.

### `AraBERT.ipynb`

AraBERT fine-tuning and evaluation notebook.

### `ARABERT_README.md`

AraBERT-specific setup, execution, and methodological notes.

### `MarBERT.ipynb`

MARBERT fine-tuning and evaluation notebook.

### `MARBERT_README.md`

MARBERT-specific setup, execution, and methodological notes.

The reported supervised setup used a stratified 80/10/10 split:

```text
training:   801
validation: 100
test:       101
```

Notebook outputs should be cleared before public release if they expose dataset text or other restricted content.

## `code/McNemar/`

### `mcNemar.py`

Performs pairwise exact McNemar tests on model-level correctness outcomes.

The reported analysis:

- used paired predictions on the same 1,002 zero-shot samples;
- compared 12 evaluated models;
- produced 66 pairwise comparisons;
- used the two-sided exact binomial form of McNemar's test;
- applied Holm step-down correction;
- used adjusted `p<0.05` as the significance threshold.

The script requires prediction files that match its expected schema and path configuration.

## `data/`

Reserved location for the authorised benchmark dataset.

The public repository does not include the full text dataset.

After authorised access is obtained, the expected dataset path is:

```text
data/EmotionsFile_GoldLabelCSV.csv
```

See:

```text
docs/DATASET.md
```

for the required columns, class distribution, integrity checks, privacy conditions, and authorised-use guidance.

## `docs/`

Contains repository-level documentation.

### `ACKNOWLEDGEMENTS.md`

Acknowledges contributors, institutions, software providers, and research support.

### `CODE_OF_CONDUCT.md`

Expected standards of conduct for repository participation.

### `COHEN_KAPPA.md`

Documents the inter-annotator agreement calculation and interpretation.

### `CONTRIBUTING.md`

Contribution rules for code, documentation, generated outputs, and methodological changes.

### `DATASET.md`

Authoritative public documentation for:

- dataset size;
- label set;
- final class distribution;
- annotation agreement;
- authorised placement;
- integrity checks;
- privacy and licensing constraints.

### `EVALUATION_PROTOCOL.md`

Concise description of the benchmark evaluation logic.

### `PROJECT_STRUCTURE.md`

This file.

### `REPRODUCIBILITY.md`

Detailed guidance for re-running the evaluation pipeline and documenting replication differences.

## `output/`

Reserved for files generated by new executions of the repository code.

The original sample-level outputs used in the reported study are not included in the public repository.

### `output/one_vote/`

Reserved for newly generated zero-shot single-pass outputs.

### `output/few_shot/`

Reserved for newly generated few-shot single-pass outputs.

### `output/three_votes/`

Reserved for newly generated three-vote outputs.

### `output/GPT_runs/`

Reserved for newly generated GPT-4o repeated-run outputs.

Generated files should be placed in clearly named subdirectories, for example:

```text
predictions/
metrics/
figures/
logs/
```

New results must not be described as the original archived results.

## Dataset and Output Policy

The repository intentionally separates:

- code;
- documentation;
- restricted dataset material;
- newly generated outputs.

The public software release must not contain:

- API credentials;
- private data;
- usernames or platform identifiers;
- unauthorised social-media text;
- notebook outputs that reveal restricted dataset rows;
- claims that original prediction outputs are included when they are not.

## Recommended Execution Order

A complete re-run should proceed in this order:

1. configure the Python environment;
2. place the authorised dataset in `data/`;
3. configure provider API keys locally;
4. verify model identifiers and access;
5. run zero-shot single-pass scripts;
6. run few-shot single-pass scripts;
7. run three-vote scripts at `T=0.3`;
8. run GPT-4o stability analysis if required;
9. run the supervised baseline notebooks;
10. verify prediction schemas and output locations;
11. run the McNemar analysis;
12. compare newly generated aggregate values with the manuscript.

## Important Reproducibility Note

Hosted models and provider APIs may change over time.

Differences may arise because of:

- model alias updates;
- backend serving changes;
- provider-side nondeterminism;
- parameter support differences;
- safety filtering;
- API availability;
- rate limits;
- silent model revisions.

Therefore, newly generated files should be described as re-runs or replications, not as the original experimental outputs.
