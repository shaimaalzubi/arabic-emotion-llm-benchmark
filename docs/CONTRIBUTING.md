# Contributing

Thank you for your interest in contributing to this research software repository.

This repository accompanies a scientific benchmark. Contributions must preserve methodological transparency, data integrity, reproducibility, and the traceability of experimental changes.

## Types of Contributions

Appropriate contributions include:

- correcting documentation;
- fixing reproducibility issues;
- improving error handling without altering the documented method;
- adding tests or validation checks;
- clarifying dataset access or preprocessing;
- adding a clearly separated replication or extension;
- reporting a bug in a script or metric calculation;
- improving accessibility and code readability.

Changes that alter prompts, models, inference settings, labels, preprocessing, metric definitions, or generated outputs must be documented explicitly.

## Before Contributing

Please review:

- [`README.md`](README.md)
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- [`DATASET.md`](DATASET.md)
- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

Check that your proposed change does not expose:

- API keys;
- `.env` files;
- credentials;
- confidential information;
- personal data;
- restricted datasets;
- provider response data that cannot legally be redistributed.

## Reporting an Issue

When reporting a problem, include:

- the affected file or script;
- the operating system;
- the Python version;
- the package versions;
- the model and provider, where relevant;
- the exact command used;
- the expected behaviour;
- the observed behaviour;
- the complete error message with secrets removed;
- whether the issue affects reported or newly generated results.

Do not paste API keys, tokens, personal data, or confidential content into an issue.

## Proposing a Change

Use a focused branch and keep each contribution limited to one logical change.

Suggested branch names:

```text
docs/clarify-reproduction
fix/metric-calculation
feature/add-replication
```

Use clear commit messages, for example:

```text
Clarify dataset path in reproduction guide
Fix unknown-label count in summary metrics
Add separate replication for updated model
```

## Development Setup

Clone the repository:

```bash
git clone https://github.com/shaimaalzubi/Benchmark.git
cd Benchmark
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file only when required for API-based experiments. Never commit it.

## Code Requirements

Contributed code should:

- use clear, descriptive file and variable names;
- preserve UTF-8 encoding for Arabic text;
- validate required dataset columns;
- validate the permitted label set;
- avoid hard-coded API credentials;
- use relative or configurable paths where practical;
- create output directories safely;
- document model identifiers and inference settings;
- distinguish provider failures from model predictions;
- preserve raw outputs when required for auditability;
- include concise comments for non-obvious logic.

## Experimental Integrity

A contribution must not silently change:

- dataset samples or labels;
- sample count;
- row order when order affects execution;
- prompt examples;
- system prompts;
- model identifiers;
- temperature;
- vote count;
- tie-break policy;
- retry behaviour;
- unknown-label handling;
- metric definitions;
- previously generated output files.

When one of these changes is necessary:

1. explain the reason;
2. document the old and new settings;
3. store new outputs separately;
4. update the relevant documentation;
5. add an entry to [`CHANGELOG.md`](CHANGELOG.md);
6. state whether the associated paper results are affected.

## Generated Results

Do not overwrite prediction, metric, or figure files from a prior execution without documenting the change.

New executions should be placed in a clearly named folder and described as one of:

- reproduction;
- replication;
- extension;
- correction;
- re-run.

Any result that differs from the values reported in the associated manuscript must be documented clearly.

## File Placement

Follow [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

In general:

- project-wide documentation belongs in the repository root;
- shared benchmark data belongs in `data/`;
- experiment code belongs in the corresponding `code/` folder;
- sample-level outputs belong in `output/predictions/`;
- evaluation summaries belong in `output/metrics/`;
- figures belong in `output/figures/`.

## Pull Request Checklist

Before submitting a contribution, confirm:

- [ ] The change has one clear purpose.
- [ ] No secrets or personal data are included.
- [ ] Python files run without syntax errors.
- [ ] Dataset and output paths are documented.
- [ ] Experimental settings are preserved or explicitly changed.
- [ ] New outputs do not overwrite prior generated outputs without documentation.
- [ ] Documentation has been updated.
- [ ] `CHANGELOG.md` has been updated when appropriate.
- [ ] The contribution follows the Code of Conduct.

## Licence and Attribution

By contributing, you agree that your contribution may be distributed under the repository's MIT License.

You must have the right to submit all contributed code, documentation, data, and media. Do not contribute third-party material unless its licence permits inclusion and all required attribution is provided.

## Research Credit

Substantial scientific contributions may require acknowledgement, citation, or authorship discussion under the policies governing the associated research project and publication. A code contribution alone does not automatically establish authorship.

Questions about academic credit should be resolved with the project leads before substantial unpublished work is contributed.
