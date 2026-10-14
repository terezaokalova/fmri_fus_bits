# fmri_fus_bits

**Question:** How many bits per second do hemodynamic signals carry about a task, how does
that rate depend on task design and acquisition, and how much of the gap between
functional ultrasound (fUS) and fMRI comes from spatial resolution, temporal sampling,
the hemodynamic kernel and noise?

This repository contains data access with recorded provenance, explicit observation
transforms, and information-rate estimators. Research decisions, provisional results and
interpretations are recorded in the ignored `local/RESEARCH_LOG.md`. Numerical
experiments are reproducible from their code and run manifests.

## Data

| Key | Modality | Source | Terms |
|---|---|---|---|
| `griggs2023_bmi` | fUS, macaque posterior parietal cortex, memory-guided saccades and reaches | [CaltechDATA 10.22002/pa710-cdn95](https://doi.org/10.22002/pa710-cdn95) | CC BY-NC 4.0 |
| `nunez_elizalde2022` | fUS with simultaneous Neuropixels, awake mouse | [figshare 10.6084/m9.figshare.19316228.v1](https://doi.org/10.6084/m9.figshare.19316228.v1) | CC BY 4.0 |
| `hcp_ya_motor` | 3T fMRI, human motor task | `s3://hcp-openaccess/HCP_1200` | [HCP Open Access Data Use Terms](https://balsa.wustl.edu/project?project=HCP_YA) |

Sources are pinned in `configs/sources.toml`. The two fUS archives are read selectively
over HTTP range requests; single sessions are extracted, never whole archives. HCP access
requires your own credentials from BALSA (Get/Reset AWS S3 Access), stored as the AWS
profile `hcp`. Downloaded recordings stay in the ignored `data/` directory and are not
redistributed.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked
```

This creates `.venv` and installs the package in editable mode. `uv.lock` records exact
dependencies. Select `.venv/bin/python` as the notebook kernel.

## First runs

```sh
uv run python scripts/probe_zips.py
uv run pytest
bash scripts/check.sh
```

## Layout

```text
src/fmri_fus_bits/   reusable, importable Python code
  sources.py         typed records for configs/sources.toml; no network access
  information.py     mutual information of Gaussian class constellations; crossnobis
  noise.py           white plus slow noise in trial-locked windows, cross-fitted
  channel.py         trial-locked channel: response shape, amplitudes, Bayes classifier
  kernels.py         hemodynamic kernels as step responses; gamma kernel fit
  equalizers.py      per-trial information under a trial design: zero-forcing,
                     interference as noise, decision feedback
scripts/             repeatable entry points; no scientific logic duplicated here
configs/             small, versioned source and experiment settings
tests/               mathematical identities and data-integrity checks, offline
data/                ignored recordings, manifests and caches
results/             ignored generated plots, tables and run manifests
local/               ignored research log and handoffs
```

Modules are added with the code that needs them. Loaders handle dataset conventions and
record provenance; estimators accept arrays and never download data. Import contracts in
`pyproject.toml` enforce these boundaries.

## Development workflow

Single contributor: commits go directly to `main`, one feature or layer per commit, about
200 changed source lines and at most about 500. Generated lockfiles are reviewed
separately from the source-line budget.

Commit messages use `type(scope): subject`, with types `feat`, `fix`, `refactor`, `docs`,
`chore` and `test`, and a repository area such as `core`, `datasets`, `information`,
`scripts` or `project`. Use an imperative, lowercase subject with no final period. Messages
are plain ASCII and contain no attribution.

Before every commit, run `bash scripts/check.sh`. Its five gates are `ruff format`,
`ruff check --fix`, `mypy --strict`, `lint-imports` and `pytest`. Strict typing covers
`src/` and `scripts/`. Only `remotezip`, which ships without type information, has an
import-level typing exception.

Tracked Markdown documents describe implementation and usage. Research notes and
provisional interpretations stay in the ignored local log.
