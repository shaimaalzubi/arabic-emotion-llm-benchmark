# Synthetic Arabic Emotion Sample

## Overview

This directory includes a small synthetic sample created solely to demonstrate the expected dataset structure used in the Arabic emotion-classification benchmark.

The file:

```text
synthetic_example_50.csv
```

contains 50 manually generated Arabic sentences distributed across the five emotion classes used in the study.

## Class Distribution

| Emotion | Number of Samples |
|---|---:|
| Anger | 10 |
| Joy | 10 |
| Sadness | 10 |
| Fear | 10 |
| Neutral | 10 |
| **Total** | **50** |

## File Structure

The synthetic file contains the following columns:

| Column | Description |
|---|---|
| `sample_id` | Unique identifier assigned to each synthetic example |
| `text` | Manually generated Arabic sentence |
| `emotion` | Assigned emotion label |
| `source` | Indicates that the example was synthetically generated |
| `is_synthetic` | Boolean flag confirming that the row is synthetic |

## Allowed Labels

The `emotion` column uses the same five-class label space as the benchmark:

```text
anger
joy
sadness
fear
neutral
```

## Purpose

The synthetic sample is provided to:

- illustrate the expected CSV structure;
- demonstrate the required column names;
- support testing of code paths and data-loading functions;
- provide examples of valid emotion labels;
- allow users to inspect the repository without accessing the restricted full-text dataset.

## Important Notice

The synthetic examples are not part of the benchmark dataset.

They were manually created for documentation and demonstration purposes and were not collected from Twitter, YouTube, or any other social-media platform.

The synthetic file must not be used to reproduce the reported benchmark metrics, McNemar results, model rankings, confusion matrices, or supervised baseline results.

## Relationship to the Original Dataset

The original benchmark contains 1,002 manually curated Arabic sentences derived primarily from publicly accessible social-media content.

The full text dataset is not included in the public repository because verbatim social-media text may remain searchable and potentially re-identifiable even after usernames, hyperlinks, timestamps, and platform-specific metadata are removed.

Access to the original dataset may be provided by the corresponding author upon reasonable request, subject to applicable ethical, institutional, and data-use requirements.

## Using the Synthetic File

The synthetic file may be loaded using pandas:

```python
import pandas as pd

df = pd.read_csv(
    "data/synthetic_example_50.csv",
    encoding="utf-8-sig",
)

print(df.head())
print(df["emotion"].value_counts())
```

Expected distribution:

```text
anger      10
joy        10
sadness    10
fear       10
neutral    10
```

## Validation Example

```python
allowed_labels = {
    "anger",
    "joy",
    "sadness",
    "fear",
    "neutral",
}

assert len(df) == 50
assert set(df["emotion"]) == allowed_labels
assert df["sample_id"].is_unique
assert df["text"].notna().all()
assert df["is_synthetic"].all()
```

## Citation and Use

The synthetic sample may be used for repository inspection, educational demonstrations, software testing, and format validation.

It should not be cited or described as the dataset used to obtain the scientific results reported in the associated manuscript.
