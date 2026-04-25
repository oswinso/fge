<div align="center">

# Feasibility-Guided Exploration (FGE)

</div>

<div align="center">
    <img src="/media/teaser_main.gif" />
</div>

<div align="center">

Jax official implementation of ICLR2026 paper: [Oswin So](https://oswinso.xyz/), [Eric Yang Yu](https://ericyangyu.github.io/), [Songyuan Zhang](https://syzhang092218-source.github.io), [Matthew Cleaveland](https://www.linkedin.com/in/matthew-cleaveland-4775abba/), [Mitchell Black](https://www.blackmitchell.com/), and [Chuchu Fan](https://chuchu.mit.edu): "[Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](https://oswinso.xyz/fge)".

[Webpage](https://oswinso.xyz/fge/) •
[arXiv](https://arxiv.org/abs/2602.15817) •
[Paper](https://arxiv.org/pdf/2602.15817) &ensp; ❘ &ensp;
[Setup](#Setup) •
[Quickstart](#Quickstart) •
[Citation](#Citation)

</div>

## Setup

FGE requires Python 3.12 or newer.

Create and activate a virtual environment from the repository root:

```bash
git clone https://github.com/oswinso/fge.git
cd fge
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Install the JAX backend for your machine before installing FGE:

```bash
# CPU
python -m pip install -U jax==0.8.0

# NVIDIA GPU, CUDA 12
python -m pip install -U "jax[cuda12]==0.8.0"
```

Install the PyTorch CPU wheel before installing FGE:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

Then install FGE in editable mode:

```bash
python -m pip install -e .
```

The editable install pulls runtime dependencies from `pyproject.toml`, including public HTTPS GitHub dependencies. It does not require GitHub SSH access.

To run the F-16 environments or scripts, install the optional F-16 dependencies:

```bash
python -m pip install -e ".[f16]"
```

`requirements-lock.txt` is a known-good environment snapshot. It is useful for debugging or reproducing the original development environment, but it is not the recommended first install path for new users.

## Quickstart

Run the Domain Randomization baseline entrypoint for an environment from the repository root. These commands use the paper-default hyperparameters.

```bash
# ToyLevels
python scripts/toylevels/run_ppo.py

# Dubins Circle
python scripts/dub_circ/run_ppo.py

# F-16 Avoid
python scripts/f16/run_ppo.py

# Kinetix Lander
python scripts/lander/run_ppo.py

# Kinetix Lander Hard
python scripts/lander_hard/run_ppo.py

# MuJoCo Hopper
python scripts/mujoco/hopper/run_ppo.py

# MuJoCo Cheetah
python scripts/mujoco/cheetah/run_ppo.py
```

The F-16 command requires the optional F-16 dependencies:

```bash
python -m pip install -e ".[f16]"
```

Weights & Biases logging is disabled by default. Pass `--use-wandb` to a run script to enable it.

Implemented algorithms:

| Script suffix | Algorithm |
| --- | --- |
| `run_ppo_fge.py` | [Feasibility-Guided Exploration (FGE)](https://arxiv.org/pdf/2602.15817), **Ours** |
| `run_ppo.py` | [Domain Randomization](https://arxiv.org/abs/1703.06907) |
| `run_ppo_plr.py` | [Prioritized Level Replay (PLR)](https://arxiv.org/abs/2010.03934) |
| `run_ppo_sfl.py` | [Sampling for Learnability (SFL)](https://arxiv.org/pdf/2408.15099) |
| `run_ppo_vds.py` | [Value Disagreement based Sampling (VDS)](https://arxiv.org/pdf/2006.09641) |
| `run_ppo_paired.py` | [PAIRED](https://arxiv.org/pdf/2012.02096) |
| `run_ppo_accel.py` | [ACCEL](https://arxiv.org/pdf/2203.01302) |
| `run_ppo_rarl.py` | [Robust Adversarial Reinforcement Learning (RARL)](https://arxiv.org/pdf/1703.02702) |
| `run_ppo_farr.py` | [Feasible Adversarial Robust Reinforcement Learning for Underspecified Environments (FARR)](https://arxiv.org/pdf/2207.09597) |

## Citation
If you use FGE in your research, please cite the original paper:

```bibtex
@inproceedings{so2026solving,
  title     = {Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning},
  author    = {So, Oswin and Yu, Eric Yang and Zhang, Songyuan and Cleaveland, Matthew and Black, Mitchell and Fan, Chuchu},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026},
}
