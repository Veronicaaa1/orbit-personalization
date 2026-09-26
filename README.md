# Data-Topology Compatibility Enables Personalization in Decentralized Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code accompanying the paper **"Data-Topology Compatibility Enables Personalization in Decentralized Learning"** by Veronica Xu, Lingfei Wang, Chenlei Leng, Fengxiang He.

## Overview

This repository provides numerical experiments, MNIST/CIFAR-10 experiments with MLP-LN and ResNet20-GN models, and tools to verify supplied experimental results and regenerate the paper's figures and tables.

## Installation

### Prerequisites

Use Python **3.10 or later** with pip. Dependencies are listed in [requirements.txt](real_world_experiments/requirements.txt). Verification and plotting can run on CPU. Training supports CPU or CUDA with a compatible PyTorch installation.

### Steps

1. Download and extract the code archive.
2. Open a terminal in the project root and activate your Python environment.
3. Install the dependencies and local package in editable mode:

   ```bash
   python -m pip install -r real_world_experiments/requirements.txt
   python -m pip install -e real_world_experiments
   ```

4. Download the complete results archive ([`results.zip`](real_world_experiments/results.zip)), and save it in `real_world_experiments/`. The archive is stored using Git LFS. If you cloned the repository, you can instead retrieve it by installing Git LFS and running `git lfs install` followed by `git lfs pull` from the project root. Skip this step if the complete `real_world_experiments/results/` directory is already available. The ZIP must contain a top-level `results/` directory. Run the following command from the project root:

   ```bash
   python -B -m zipfile -e real_world_experiments/results.zip real_world_experiments
   ```

   After extraction, confirm that `real_world_experiments/results/SHA256_MANIFEST.csv` exists.

Run all subsequent commands from the project root.

The results archive is required for real-data training, archive verification, and regenerating real-data figures and tables. Installation and numerical experiments do not require it. Keep the archive contents unchanged.

## Run Experiments

### Numerical Experiments

Run the numerical simulations, including the nonquadratic experiments, and generate their figures and data:

```bash
python -B "numerical experiments/compatibility_experiments.py" --out reproduction/numerical --nonquadratic
```

Outputs are saved under `reproduction/numerical/`, with figures in `figures/` and simulation data in `data/`. The output directory must be new or empty.

### Real-Data Experiments

Run all four combinations of MNIST/CIFAR-10 and MLP-LN/ResNet20-GN, using conditions `main`, `no_communication`, `misaligned_data` and `fully_symmetric`, with formal seeds **1, 2, 3**. The workflow includes data preparation, pilot runs, communication-coefficient selection, formal training, verification and analysis. Parameters are in [reproduce_final.yaml](real_world_experiments/configs/reproduce_final.yaml).

```bash
python -B real_world_experiments/scripts/reproduce_realworld.py --stage all --execute --download
```

Training results are saved to `reproduction/real_world/outputs/`; analysis results are saved to `reproduction/real_world/reports/`. Internet access is needed to download datasets that are not already available.

### Regenerate Figures and Tables

Regenerate real-data Figure 2, Figures 8–9 and Tables 2–6 from the supplied experiment archive. This requires no new training, raw datasets or paper PDF:

```bash
python -B real_world_experiments/scripts/reproduce_paper.py --output real_world_experiments/paper_results
```

Outputs are saved to `real_world_experiments/paper_results/`, with PDF/PNG figures in `figures/`, CSV tables in `tables/` and plotting data in `data/`. The output directory must be new or empty.

## Verify Supplied Results

```bash
python -B real_world_experiments/scripts/verify_export.py
```

This checks the supplied archive's integrity and result consistency and prints a JSON report in the terminal.

## Project Structure

| Path | Contents |
| --- | --- |
| [numerical experiments/](numerical%20experiments/) | Numerical simulation and plotting code |
| [real_world_experiments/src/](real_world_experiments/src/) | Data preparation, models, training and analysis |
| [real_world_experiments/configs/](real_world_experiments/configs/) | Experiment configurations |
| [real_world_experiments/scripts/](real_world_experiments/scripts/) | Training, verification and figure-generation entry points |
| `real_world_experiments/results/` | Reduced result archive for all four combinations |
| [real_world_experiments/tests/](real_world_experiments/tests/) | Test source files |

