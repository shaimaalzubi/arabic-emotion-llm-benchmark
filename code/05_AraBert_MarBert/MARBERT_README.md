# MARBERT Fine-Tuning Baseline

## Overview

This notebook fine-tunes **MARBERT** for five-class Arabic emotion classification.

- Base model: `UBC-NLP/MARBERT`
- Task: single-label sequence classification
- Framework: Hugging Face Transformers
- Dataset size: 1,002 Arabic sentences
- Labels: `anger`, `fear`, `joy`, `neutral`, `sadness`
- Random seed: `42`

## Public-Release Cleaning

The public `MarBERT.ipynb` has been cleaned as follows:

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
correct
```

## Dataset and Split

The notebook reads the authorised dataset from:

```text
/content/EmotionsFile_GoldLabelCSV.csv
```

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

## Model Configuration

```text
MODEL_NAME = "UBC-NLP/MARBERT"
MAX_LEN = 128
SEED = 42
```

## Training Configuration

| Parameter | Value |
|---|---:|
| Learning rate | `2e-5` |
| Training batch size | `4` |
| Evaluation batch size | `4` |
| Epochs | `1` |
| Weight decay | `0.01` |
| Evaluation during training | Disabled |
| Save strategy | Every epoch |
| Best-model loading | Disabled |
| Saved checkpoints retained | `1` |
| Random seed | `42` |

## Reported Results

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Validation | 0.7500 | 0.7556 | 0.7515 |
| Test | 0.7921 | 0.7847 | 0.7915 |

## Repository Placement

```text
code/05_AraBert_MarBert/
├── MarBERT.ipynb
└── MARBERT_README.md
```

## Running the Notebook

1. Open `MarBERT.ipynb` in a clean Google Colab runtime.
2. Select a GPU runtime where available.
3. Obtain authorised access to the dataset.
4. Upload the dataset to the configured path.
5. Run all cells in order.
6. Confirm the 801/100/101 stratified split.
7. Save only privacy-preserving predictions without the source text.
8. Do not commit model checkpoints, credentials, or restricted dataset content.

Exact metrics may vary slightly across hardware, CUDA, PyTorch, and Transformers versions.
