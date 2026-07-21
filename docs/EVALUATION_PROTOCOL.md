# Evaluation Protocol

## Overview

This document describes the evaluation procedures used in:

**Deployment-Oriented Evaluation of Large Language Models for Arabic Emotion Classification**

The benchmark evaluates sentence-level Arabic emotion classification across five valid labels:

```text
anger
joy
sadness
fear
neutral
```

The special label:

```text
unknown
```

is not a gold emotion class. It is an evaluation outcome used when a model response cannot be mapped to one of the five valid labels.

## Evaluation Scope

The study includes:

1. zero-shot single-pass evaluation;
2. few-shot single-pass evaluation;
3. three-vote self-consistency evaluation;
4. GPT-4o run-to-run stability analysis;
5. supervised AraBERT and MARBERT baselines;
6. pairwise exact McNemar significance testing.

## Reference Labels

The benchmark contains 1,002 samples.

The final adjudicated reference label for each sample is stored in:

```text
emotion
```

All model predictions are evaluated against this field.

## Primary Zero-Shot Protocol

The primary benchmark uses:

- one inference request per sample;
- single-pass inference;
- `k=1`;
- zero-shot prompting;
- the same five-label target space;
- a harmonised task specification;
- a common target JSON structure;
- temperature set to zero where the provider interface accepts an explicit temperature parameter;
- provider-specific structured-output, parsing, retry, and request-management procedures where required.

The target response structure is:

```json
{ "label": "<emotion>" }
```

The primary benchmark is evaluated on all 1,002 samples.

## Few-Shot Protocol

The few-shot setting uses:

- five manually constructed demonstrations;
- one demonstration for each emotion class;
- demonstrations external to the 1,002-sample evaluation dataset;
- one inference request per sample;
- `k=1`;
- temperature set to zero where explicitly supported;
- the same valid label set and final output-validation criteria as the primary benchmark.

The five demonstrations are included in the prompt without changing model parameters.

Provider-specific prompt wording may differ where required by API interfaces, but the task, label space, and expected final output remain harmonised.

## Three-Vote Self-Consistency Protocol

The three-vote protocol is a separate complementary analysis.

It uses:

- three independent generations per sample;
- sampling temperature `T=0.3` where explicitly supported;
- separate validation of every generated response;
- valid-vote aggregation;
- a final valid label only when at least two of the three votes agree.

The three-vote results are reported separately from the primary single-pass benchmark because:

- the decoding strategy differs;
- the protocol introduces stochastic sampling;
- the inference cost is approximately three times higher;
- the final prediction depends on vote aggregation.

## Vote-Level Validation

Each generated response is first mapped to either:

```text
anger
joy
sadness
fear
neutral
unknown
```

A vote becomes `unknown` when the response is non-evaluable.

Examples include:

- malformed JSON;
- incomplete JSON;
- missing label field;
- unsupported label;
- empty response;
- blocked response;
- API or HTTP failure;
- parsing failure;
- output truncation;
- exhausted retries.

Provider-specific technical failure categories may differ, but the final label-validation rules remain aligned.

## Three-Vote Aggregation Rules

### Majority

When at least two valid votes agree on the same emotion label, that label becomes the final prediction.

Example:

```text
joy, joy, sadness -> joy
```

Reason:

```text
majority
```

### All Failed

When all three votes are non-evaluable:

```text
unknown, unknown, unknown -> unknown
```

Reason:

```text
all_failed
```

### Three Different Valid Labels

When all three votes are valid but different:

```text
joy, sadness, fear -> unknown
```

Reason:

```text
all_different_111
```

### No Two-Vote Majority

When the available valid votes do not produce a two-vote majority because one or more votes are non-evaluable:

```text
joy, sadness, unknown -> unknown
```

or:

```text
joy, unknown, unknown -> unknown
```

Reason:

```text
no_majority_11u
```

Intermediate `unknown` votes are excluded before counting valid labels, but a final valid label is retained only when it receives at least two votes.

## Unknown Output Definition

`unknown` is an operational evaluation category.

It should not be interpreted automatically as:

- explicit abstention;
- calibrated uncertainty;
- refusal;
- lack of confidence.

It may arise from either:

1. a model-level response that does not match the allowed label space; or
2. an implementation-level or API-level failure that prevents evaluation.

Therefore, unknown rate measures the frequency of non-evaluable final outputs under the implemented inference pipeline.

## Strict Evaluation

Strict evaluation retains all 1,002 samples.

A final prediction is scored as correct only when:

```text
prediction == reference label
```

A final `unknown` prediction is treated as incorrect.

Strict accuracy is:

```text
number of correct predictions / total number of samples
```

Strict macro-F1 is calculated over the full benchmark while treating `unknown` predictions as incorrect outcomes.

Strict evaluation reflects settings where a valid prediction is expected for every input.

## Covered Evaluation

Covered evaluation excludes samples whose final prediction is:

```text
unknown
```

Predictive metrics are then calculated only on samples with valid emotion predictions.

Covered evaluation measures classification performance conditional on producing an evaluable label.

Covered metrics must not be interpreted without also reporting coverage or unknown rate.

## Coverage

Let:

```text
N = total number of samples
N_valid = number of final predictions in the five-label set
```

Coverage is:

```text
Coverage = N_valid / N
```

## Unknown Rate

Let:

```text
N_unknown = number of final unknown predictions
```

Unknown rate is:

```text
Unknown Rate = N_unknown / N
```

Because every final output is either valid or unknown:

```text
Coverage = 1 - Unknown Rate
```

## Predictive Metrics

The benchmark reports, where applicable:

- accuracy;
- macro-F1;
- weighted F1;
- covered accuracy;
- covered macro-F1;
- coverage;
- unknown rate.

Macro-F1 is emphasised because:

- the task is multiclass;
- the class distribution is moderately imbalanced;
- each emotion class contributes equally to the unweighted class average.

Micro-F1 is not required as a headline metric because, in single-label multiclass classification, it is generally equivalent to accuracy.

## Supervised Baseline Evaluation

AraBERT and MARBERT are evaluated separately as supervised contextual reference models.

The dataset split is stratified:

```text
training:   801
validation: 100
test:       101
```

Reported baseline metrics are calculated on the held-out test set.

These scores are not treated as strictly matched comparisons with the zero-shot LLM benchmark because:

- the supervised models are trained on benchmark-labelled samples;
- the LLMs are not fine-tuned on the benchmark;
- the supervised models are evaluated on 101 held-out samples;
- the LLMs are evaluated on all 1,002 samples.

## Run-to-Run Stability Protocol

GPT-4o is evaluated across three complete zero-shot runs using:

- the same 1,002 samples;
- the same prompt template;
- `k=1`;
- temperature zero;
- aligned parsing and validation procedures.

The analysis reports:

### Consistency

The proportion of samples receiving the same predicted label in all three runs.

### Variability

The proportion of samples whose prediction changes in at least one run.

### Persistent Error Rate

The proportion of samples misclassified in all three runs.

### Correctness-Change Rate

The proportion of samples that are correct in at least one run and incorrect in at least one other run.

Temperature zero reduces user-controlled sampling variability but does not guarantee identical API outputs across repeated executions.

## McNemar Significance Testing

Pairwise exact McNemar tests are applied to primary zero-shot predictions.

For each model pair:

- predictions must refer to the same 1,002 samples;
- each sample is converted into a binary correctness outcome;
- `unknown` is treated as incorrect;
- discordant correctness counts are computed.

Let:

```text
b = samples where Model A is correct and Model B is incorrect
c = samples where Model A is incorrect and Model B is correct
```

The analysis uses:

- the two-sided exact binomial McNemar test;
- all 66 pairwise comparisons among 12 models;
- Holm step-down correction;
- adjusted significance threshold `p<0.05`.

## Output Validation Principles

Across hosted-model experiments, the benchmark aligns:

- target emotion labels;
- final definition of a valid prediction;
- final definition of a non-evaluable prediction;
- strict and covered scoring rules.

The benchmark does not assume that providers expose identical:

- structured-output mechanisms;
- request schemas;
- retry behaviour;
- rate limits;
- timeout handling;
- token limits;
- parsing requirements.

These provider-specific implementation differences must be documented because they may influence operational outcomes.

## Required Prediction Fields

A reproducible prediction file should contain, at minimum:

```text
sample_id or dataset_index
gold_label
prediction
model
provider
run_id
```

For three-vote experiments, it should additionally contain:

```text
vote_1
vote_2
vote_3
final_prediction
final_reason
```

Where available, preserve:

```text
temperature_requested
temperature_sent
technical_failure_reason
timestamp
prompt_version
parser_version
```

Restricted dataset text should not be included in public outputs unless redistribution is explicitly authorised.

## Interpretation Boundaries

The evaluation supports conclusions about:

- predictive performance;
- output validity;
- coverage;
- unknown rate;
- structured-output compliance;
- repeated-output consistency;
- observable error patterns.

It does not directly establish:

- probability calibration;
- causal reasons for model errors;
- internal reasoning quality;
- latent uncertainty;
- general robustness across all Arabic dialects;
- provider-level superiority;
- reliability in every machine-learning sense.

The term operational reliability refers specifically to observable deployment-oriented behaviour under the implemented evaluation protocol.
