# Cohen's Kappa and Annotation Agreement

## Overview

This document records the inter-annotator agreement analysis for the Arabic emotion classification dataset used in:

**Deployment-Oriented Evaluation of Large Language Models for Arabic Emotion Classification**

The dataset contains:

```text
1,002 Arabic sentences
```

Each sentence was independently labelled by two human annotators before adjudication.

## Emotion Labels

The annotation task used five labels:

```text
anger
joy
sadness
fear
neutral
```

Each annotator assigned one dominant emotion label to every sentence.

## Agreement Counts

Across all 1,002 samples:

```text
agreements:    980
disagreements: 22
total:         1,002
```

The 22 disagreement cases were resolved through adjudication to establish the final reference label stored in the `emotion` column.

The adjudicated labels were not used to calculate Cohen's kappa.

## Observed Agreement

Observed agreement is:

```text
P_o = agreements / total samples
```

Therefore:

```text
P_o = 980 / 1002
P_o = 0.9780
```

## Expected Agreement

Expected agreement was calculated from the marginal label distributions of the two independent annotators:

```text
P_e = Σ P_rater1(c) × P_rater2(c)
```

where `c` ranges over:

```text
anger
joy
sadness
fear
neutral
```

The resulting expected agreement was:

```text
P_e = 0.2094
```

## Cohen's Kappa

Cohen's kappa is:

```text
κ = (P_o - P_e) / (1 - P_e)
```

Substituting the observed values:

```text
κ = (0.9780 - 0.2094) / (1 - 0.2094)
κ = 0.9722
```

## Final Reported Value

The reported agreement statistic is:

```text
Cohen's κ = 0.9722
```

Approximately:

```text
2.20%
```

of samples required adjudication.

## Interpretation

A kappa value of `0.9722` indicates very high agreement between the two annotators under the specified five-class annotation protocol.

However, this value should be interpreted in the context of the dataset-curation process.

During curation:

- highly incomplete sentences were excluded;
- severely ambiguous sentences were excluded;
- retained samples were required to contain a sufficiently interpretable dominant emotional signal;
- a single-label annotation scheme was used.

These decisions likely increased annotation consistency.

The result should not be interpreted as evidence that Arabic emotional expression is generally unambiguous.

Naturally occurring Arabic social-media content may contain:

- emotional blending;
- pragmatic ambiguity;
- dialectal variation;
- irony;
- sarcasm;
- incomplete context;
- culturally specific expressions;
- multiple plausible labels.

The reported kappa therefore reflects agreement within this curated dataset and this annotation protocol.

## Relationship to Final Reference Labels

The annotation workflow was:

1. Annotator 1 assigned an independent label.
2. Annotator 2 assigned an independent label.
3. Cohen's kappa was calculated from the two independent label sets.
4. The 22 disagreement cases were reviewed.
5. A final adjudicated label was stored in `emotion`.
6. All experiments used `emotion` as the reference label.

For the 980 agreement cases, the final reference label retained the shared annotation.

For the 22 disagreement cases, the final reference label reflected the adjudicated decision.

## Reproduction Requirements

To reproduce the kappa calculation exactly, the following are required:

```text
annotator_1_label
annotator_2_label
```

for all 1,002 samples before adjudication.

The final `emotion` column alone is not sufficient to reproduce Cohen's kappa because it contains adjudicated labels.

## Example Calculation

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(
    annotator_1_labels,
    annotator_2_labels
)

print(kappa)
```

Expected result:

```text
0.9722
```

Small differences in the final displayed decimal may occur depending on floating-point formatting.

## Integrity Checks

Before reproducing the statistic, verify:

- both annotation arrays contain 1,002 labels;
- neither array contains missing values;
- every label belongs to the five-class set;
- labels are aligned by the same sample identifier;
- adjudicated labels are not substituted for either annotator's original labels.

Recommended checks:

```python
valid_labels = {
    "anger",
    "joy",
    "sadness",
    "fear",
    "neutral",
}

assert len(annotator_1_labels) == 1002
assert len(annotator_2_labels) == 1002
assert set(annotator_1_labels) <= valid_labels
assert set(annotator_2_labels) <= valid_labels
```

## Reported Manuscript Values

The manuscript and repository documentation should use the following values consistently:

| Measure                |  Value |
| ---------------------- | -----: |
| Total samples          |  1,002 |
| Agreements             |    980 |
| Disagreements          |     22 |
| Observed agreement     | 0.9780 |
| Expected agreement     | 0.2094 |
| Cohen's kappa          | 0.9722 |
| Adjudicated proportion |  2.20% |

## References

The associated manuscript cites the standard sources for Cohen's kappa and its interpretation.

This repository document is intended to record the project-specific calculation and should not replace the formal methodological references in the paper.
