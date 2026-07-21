# Dataset Documentation

## Overview

This repository supports the experiments reported in:

**Deployment-Oriented Evaluation of Large Language Models for Arabic Emotion Classification**

The benchmark dataset contains:

- **1,002 manually curated Arabic text samples**
- **5 emotion classes**
- **one final adjudicated reference label per sample**
- **two independent human annotations per sample before adjudication**

The dataset was designed for sentence-level Arabic emotion classification under zero-shot, few-shot, repeated-inference, and supervised baseline settings.

## Public Availability

The full text dataset is **not included in this public repository**.

The sentences were derived from publicly accessible social-media content and may still raise privacy, platform-policy, ethical, and redistribution considerations. Authorised access may be requested from the corresponding author, subject to applicable institutional and ethical requirements.

The software licence in this repository applies to the code only and does not grant permission to redistribute the underlying text dataset.

## Authorised Dataset Placement

After obtaining authorised access, place the dataset at:

```text
data/EmotionsFile_GoldLabelCSV.csv
```

Run all experiment scripts from the repository root unless a script documents a different working-directory requirement.

## Required Columns

The evaluation pipeline requires at least the following columns:

| Column | Description |
|---|---|
| `text` | Arabic sentence to be classified |
| `emotion` | Final adjudicated reference label used in all reported experiments |

The dataset used during annotation may also contain additional fields, such as annotator-specific labels or adjudication metadata. These additional fields are not required by the main inference scripts unless explicitly documented.

## Reference Label

The `emotion` column stores the final reference label.

- For **980 samples**, both annotators assigned the same label, and that shared label was retained.
- For **22 samples**, the annotators initially disagreed, and the final label was established through adjudication.

All reported model evaluations and supervised baseline experiments used the `emotion` column as the reference label.

## Emotion Label Set

The valid benchmark labels are:

```text
anger
joy
sadness
fear
neutral
```

All labels must:

- be lowercase;
- contain no leading or trailing whitespace;
- belong to the five-label set above.

The label `unknown` is **not** a gold emotion class. It is used only during model-output evaluation to represent a response that could not be mapped to one of the five valid emotion labels.

## Final Class Distribution

| Emotion | Samples | Percentage |
|---|---:|---:|
| Anger | 251 | 25.05% |
| Neutral | 237 | 23.65% |
| Joy | 207 | 20.66% |
| Sadness | 169 | 16.87% |
| Fear | 138 | 13.77% |
| **Total** | **1,002** | **100.00%** |

The class distribution above corresponds to the final adjudicated labels in the `emotion` column.

## Annotation Agreement

Two annotators independently labelled all 1,002 samples before adjudication.

- Agreements: **980**
- Disagreements: **22**
- Observed agreement: **0.9780**
- Expected agreement: **0.2094**
- Cohen's kappa: **0.9722**

The kappa statistic was calculated from the two independently assigned annotation sets before adjudication.

The high agreement should be interpreted in light of the dataset-curation procedure. Highly incomplete and severely ambiguous samples were excluded, and retained samples were required to contain a sufficiently interpretable dominant emotional signal for single-label annotation.

## Data Sources

The dataset was manually collected from publicly accessible online platforms:

- primarily Twitter/X content;
- a smaller subset of YouTube comments.

No private content was used.

Before inclusion in the benchmark, platform-specific identifiers and directly identifying metadata were removed.

## Data Cleaning and Preprocessing

The dataset was curated using lightweight preprocessing intended to preserve semantic content.

The reported workflow included:

- removal of user mentions;
- removal of URLs;
- removal of platform-specific metadata;
- removal of duplicate entries;
- removal of non-emotional hashtags;
- exclusion of highly incomplete or severely ambiguous samples;
- retention of emojis only when they clearly conveyed emotional meaning;
- basic Arabic orthographic normalisation;
- whitespace normalisation.

The orthographic normalisation used in the project includes:

```text
إ, أ, ٱ, آ, ا  -> ا
ى              -> ي
```

Any reproduction should preserve the same dataset version and document any change to preprocessing.

## Missing Values

Rows with missing values in either required field must not be evaluated:

```text
text
emotion
```

Before running an experiment, verify that:

- the dataset contains exactly 1,002 valid rows;
- no required value is missing;
- every reference label belongs to the five-label set;
- Arabic text is preserved using UTF-8 encoding.

## Data Integrity Checks

Recommended checks before evaluation:

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

valid_labels = {"anger", "joy", "sadness", "fear", "neutral"}
assert set(df["emotion"].astype(str).str.strip().str.lower()) <= valid_labels

expected_counts = {
    "anger": 251,
    "neutral": 237,
    "joy": 207,
    "sadness": 169,
    "fear": 138,
}

actual_counts = (
    df["emotion"]
    .astype(str)
    .str.strip()
    .str.lower()
    .value_counts()
    .to_dict()
)

assert actual_counts == expected_counts
```

If the authorised file uses a different encoding, file name, or column name, document the change and update the relevant configuration explicitly.

## Intended Use

The dataset is intended for research on:

- Arabic sentence-level emotion classification;
- zero-shot large language model evaluation;
- few-shot in-context classification;
- repeated-generation and vote-aggregation analysis;
- output-validity and coverage evaluation;
- strict versus covered performance;
- run-to-run stability analysis;
- supervised Arabic encoder baselines;
- paired statistical comparison of model predictions.

## Evaluation Protocols

The dataset supports the following protocols used in the associated study:

### Zero-shot single-pass evaluation

- one inference request per sample;
- `k=1`;
- temperature set to zero where explicitly supported by the provider interface.

### Few-shot single-pass evaluation

- five external demonstrations;
- one demonstration per emotion class;
- one inference request per sample;
- `k=1`;
- temperature set to zero where explicitly supported.

### Three-vote evaluation

- three independent generations per sample;
- sampling temperature `T=0.3` where supported;
- valid-vote aggregation;
- a final emotion label is retained only when it receives at least two votes.

### GPT-4o run-to-run analysis

- three complete zero-shot runs;
- temperature set to zero;
- comparison of prediction consistency and correctness changes across runs.

### Supervised baselines

- stratified 80/10/10 split;
- 801 training samples;
- 100 validation samples;
- 101 held-out test samples.

## Unknown Output Handling

`unknown` is an evaluation label, not a dataset class.

A model output may be assigned `unknown` when it cannot be mapped to one of the five valid emotion labels because of conditions such as:

- malformed or incomplete structured output;
- missing label field;
- unsupported label;
- empty response;
- blocked response;
- API or HTTP failure;
- parsing failure;
- output truncation.

Under strict evaluation, final `unknown` predictions are treated as incorrect.

Under covered evaluation, final `unknown` predictions are excluded before predictive metrics are calculated.

## Reproducibility Requirements

The authorised benchmark file used for reproduction should remain unchanged.

Any modification to the following must be documented:

- sample text;
- row order;
- reference labels;
- preprocessing;
- removed rows;
- column names;
- label mapping;
- encoding;
- train/validation/test split.

Future reruns of hosted model APIs may not reproduce archived predictions exactly because providers may change model aliases, infrastructure, availability, or serving behaviour.

## Privacy and Ethical Considerations

The dataset should not contain:

- usernames;
- direct profile identifiers;
- post URLs;
- timestamps that enable tracing;
- account IDs;
- private messages;
- confidential information.

Although the text was collected from public sources and anonymised, public availability does not automatically imply unrestricted redistribution rights. Any access or reuse must respect applicable ethical requirements, platform terms, institutional policies, and privacy considerations.

## Licensing

The repository's MIT licence applies to the software only.

It does **not** apply automatically to:

- the underlying social-media text;
- third-party datasets;
- provider-generated model outputs;
- content governed by external platform terms.

## Citation

When using the authorised dataset with this repository, cite:

1. the associated research article;
2. this software repository;
3. the Zenodo record after DOI assignment;
4. any original data source where applicable.

Machine-readable software citation metadata are provided in:

```text
CITATION.cff
```
