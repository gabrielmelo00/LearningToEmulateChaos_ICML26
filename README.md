# Learning to Emulate Chaos: Adversarial Optimal Transport Regularization

<p align="center">
  <a href="https://arxiv.org/pdf/2604.21097"><img src="https://img.shields.io/badge/arXiv-2603.02460-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://icml.cc/virtual/2026/poster/60531"><img src="https://img.shields.io/badge/Conference-ICML%202026-4B0082?style=for-the-badge" alt="ICML 2026"></a>
</p>

Supported systems: **Lorenz 96 (L96)** — see branches `l63`, `ks`, `kolmogorov-2d` for other experiments.  
Supported training objectives: **Optimal Transport (OT / Sinkhorn)**, **WGAN**, **baseline (L2)**.

---

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if needed
uv sync
```

> **GPU note:** `pyproject.toml` defaults to PyTorch cu118. Adjust the `[[tool.uv.index]]` URL to match your CUDA version (e.g. `cu121`, `cu124`) before running `uv sync` on your cluster.

---

## Lorenz 96

### 1 — Generate data

```bash
cd l96_data_x
python generate_data.py          # full: 2000 train / 100 val / 200 test
# Quick local test:
python generate_data.py --num_of_sample 100 --val_size 100 --test_size 100 --n_workers 8
cd ..
```

### 2 — Preprocess (add noise, convert to TIFF for fast loading)

```bash
python dataloader/dataloader_l96.py
```

### 3 — Train

**Optimal Transport (Sinkhorn)**
```bash
bash experiments/OT_l96/srun.sh
```
Key hyperparameters: `--lambda_geomloss 3`, `--blur 0.02`, `--with_geomloss_kd 0`.

**Contrastive Learning**
```bash
bash experiments/CL_l96/srun.sh
```
Key hyperparameters: `--bank_size 1000`, `--T_metricL_traj_alone 0.3`.

For single-GPU development change `--nproc_per_node` to `1` in the shell script.

### 4 — Evaluate

```bash
bash experiments/OT_l96/eval.sh
bash experiments/CL_l96/eval.sh
```

Add `--eval_LE` to also compute the leading Lyapunov exponent (~2 hrs for 200 test instances).  
Ground-truth LLEs: `python eval_scripts/LE_l96.py`  
Compare results: `python eval_scripts/read_LE.py`

---

## Lorenz 63

### 1 — Generate data

```bash
bash experiments/prepare_l63_data.sh
```

### 2 — Train and evaluate

```bash
bash experiments/run_l63_three_methods.sh          # all methods

METHOD=baseline  bash experiments/run_train_l63_once.sh
METHOD=fixed_ot  bash experiments/run_train_l63_once.sh
METHOD=wgan      bash experiments/run_train_l63_once.sh

METHOD=baseline  bash experiments/run_eval_l63_once.sh
```

---

## Kuramoto–Sivashinsky

### 1 — Generate data

```bash
bash ks_data_x_single_traj/generate.sh
python dataloader/dataloader_ks.py \
  --data_path ks_data_x_single_traj/ks_single_traj_train \
  --data_path ks_data_x_single_traj/ks_single_traj_val  \
  --data_path ks_data_x_single_traj/ks_single_traj_test \
  --noisy_scale 0.3
```

### 2 — Train and evaluate

```bash
METHOD=fixed_ot bash experiments/run_train_ks_once.sh
bash experiments/submit_all_ks.sh    # full sweep
```

---

## Repository structure

```
configuration.py          — argument parser (all systems share one parser)
utils.py                  — distributed-training helpers
scripts/
  main.py                 — unified training entry point (--l96 | --kse | --l63)
  train_utils.py          — loss, LR schedule, rollout helpers
  OT_utils.py             — Sinkhorn / fixed-OT loss wrapper
  CL_utils.py             — contrastive learning utilities
  summary.py / summary_ks.py — learnable summary nets and WGAN critic
  dataloader_init.py      — system-agnostic dataloader factory
  log.py                  — output-folder creation / checkpoint naming
  cal_stats_{l96,l63,ks}.py — per-system OT feature extraction
models/
  fno_1d_new.py           — FNO operator (L96, KS)
  mlp_l63.py              — MLP operator (L63)
dataloader/
  dataloader_l96.py / dataloader_l63.py / dataloader_ks.py
eval_scripts/
  eval_l96.py / eval_l63.py / eval_ks.py
l96_data_x/               — L96 ODE solver + data generation
l63_data_x/               — L63 ODE solver + data generation
ks_data_x_single_traj/    — KS PDE solver + single-trajectory dataset utilities
experiments/              — SLURM job scripts per system and method
```

---

## Acknowledgements

This codebase builds on [roxie62/neural_operators_for_chaos](https://github.com/roxie62/neural_operators_for_chaos). We thank the authors for open-sourcing their implementation.

## Citation

```bibtex
@article{melo2026learning,
  title={Learning to Emulate Chaos: Adversarial Optimal Transport Regularization},
  author={Melo, Gabriel and Santiago, Leonardo and Lu, Peter Y},
  journal={arXiv preprint arXiv:2604.21097},
  year={2026}
}
```

## Acknowledgements
This codebase builds on [roxie62/neural_operators_for_chaos](https://github.com/roxie62/neural_operators_for_chaos). We thank the authors for open-sourcing their implementation.

## Recommended Citation
```bibtex
@article{melo2026learning,
  title={Learning to Emulate Chaos: Adversarial Optimal Transport Regularization},
  author={Melo, Gabriel and Santiago, Leonardo and Lu, Peter Y},
  journal={arXiv preprint arXiv:2604.21097},
  year={2026}
}
```
