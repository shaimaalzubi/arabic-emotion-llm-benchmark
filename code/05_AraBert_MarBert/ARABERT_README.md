# AraBERT Fine-Tuning Baseline

## Overview

This notebook fine-tunes **AraBERT v0.2** for five-class Arabic emotion classification.

- Base model observed in the original executed notebook: `aubmindlab/bert-base-arabertv02`
- Task: single-label sequence classification
- Framework: Hugging Face Transformers
- Dataset size: 1,002 Arabic sentences
- Labels: `anger`, `fear`, `joy`, `neutral`, `sadness`
- Random seed: `42`

## Public-Release Cleaning

The public `AraBERT.ipynb` has been cleaned as follows:

- all executed cell outputs were removed;
- execution counts were cleared;
- dataset previews were removed;
- notebook widget state was removed;
- exported prediction files no longer contain the original Arabic text;
- no credentials are included.

The privacy-preserving prediction file contains:

```text
dataset_index
gold_label
prediction
```

## Important Reproducibility Warning

The supplied AraBERT notebook does not contain its original import and configuration cell. Variables used before definition include:

```text
SEED
MODEL_NAME
TEXT_COL
LABEL_COL
MAX_LEN
OUTPUT_DIR
SAVE_DIR
DATA_PATH
```

The original executed output confirms:

```text
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
SEED = 42
```

The exact original value of `MAX_LEN` is not recoverable from the supplied notebook and must not be guessed or represented as the original setting.

Before describing the notebook as standalone, restore the original initialization cell from the original working copy. A template is:

```python
import os
import re
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

SEED = 42
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
TEXT_COL = "text"
LABEL_COL = "emotion"

MAX_LEN = <ORIGINAL_MAX_LEN>

OUTPUT_DIR = "/content/arabert-output"
SAVE_DIR = "/content/final_arabert_model"
DATA_PATH = "/content/EmotionsFile_GoldLabelCSV.csv"
```

## Dataset and Split

Required columns:

```text
text
emotion
```

Final class distribution:

| Label | Samples |
|---|---:|
| anger | 251 |
| neutral | 237 |
| joy | 207 |
| sadness | 169 |
| fear | 138 |
| **Total** | **1,002** |

Stratified split:

| Split | Samples |
|---|---:|
| Training | 801 |
| Validation | 100 |
| Test | 101 |

## Training Configuration

| Parameter | Value |
|---|---:|
| Learning rate | `2e-5` |
| Training batch size | `16` |
| Evaluation batch size | `16` |
| Epochs | `4` |
| Weight decay | `0.01` |
| Evaluation strategy | Every epoch |
| Save strategy | Every epoch |
| Best-model metric | `macro_f1` |
| Load best model at end | Enabled |
| Saved checkpoints retained | `2` |
| Random seed | `42` |
| Maximum sequence length | Not recoverable from supplied notebook |

## Reported Results

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 0.7000 | 0.6936 |
| Test | 0.7129 | 0.7097 |

Test weighted-F1: `0.7149`.

## Repository Placement

```text
code/05_AraBert_MarBert/
├── AraBERT.ipynb
└── ARABERT_README.md
```

## Running the Notebook

1. Restore the missing initialization cell using the original working copy.
2. Obtain authorised access to the dataset.
3. Place the dataset at the configured local path.
4. Use a GPU runtime where available.
5. Run all cells from a clean runtime.
6. Confirm the 801/100/101 stratified split.
7. Save only privacy-preserving predictions without the source text.
8. Do not commit model checkpoints, credentials, or restricted dataset content.

The cleaned notebook should not be described as fully standalone until the missing original initialization configuration is restored and verified.
