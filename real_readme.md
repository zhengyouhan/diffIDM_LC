# diffIDM_LC — Differentiable Lane-Change for Microscopic Traffic Simulation

**Differentiating through topology-switching events in IDM+MOBIL traffic simulation using saltation matrices and straight-through Gumbel estimators.**

## Motivation

Microscopic traffic simulators (IDM + MOBIL) are essential for modeling multi-lane traffic, but their discrete lane-change events break standard automatic differentiation. When a vehicle changes lanes, the leader-follower graph switches discontinuously — the dynamics operator itself changes. Naively autodiffing through this produces biased gradients.

We solve this by treating the simulator as a **hybrid dynamical system** and composing three gradient techniques:

| Component | Handles | Technique |
|-----------|---------|-----------|
| Adjoint method | Smooth IDM dynamics | JAX reverse-mode AD |
| Saltation matrix | Topology switch (dynamics jump) | Hybrid systems theory (Burden et al. 2016) |
| ST-Gumbel | Discrete lane-change decision | Straight-through estimator (Jang et al. 2017) |

**Key insight:** This problem is structurally identical to differentiating through **contact events in rigid-body simulation**. The robotics community (Dojo, Drake) has mature solutions — we bring that math to traffic.

## Quick Start

```bash
# Setup
conda create -n difflc python=3.12
conda activate difflc
pip install -r requirements.txt

# Single ablation run (5 configs, 1 seed)
cd src && python3 -u ablation.py

# Full ablation (5 configs × 5 seeds) — takes ~42h on A100
python3 -u ablation_full.py

# Visualization
python3 animate.py
```

## Documentation

- **[Problem Statement](doc/problem-statement.md)** — What we're solving and why it's hard
- **[ST-Gumbel & Saltation](doc/st-gumbel-and-saltation.md)** — The two core techniques explained
- **[Methodology](doc/methodology.md)** — Current implementation details and ablation design

## Project Structure

```
diffIDM_LC/
├── README.md                    # This file
├── doc/
│   ├── problem-statement.md     # Problem formulation
│   ├── st-gumbel-and-saltation.md  # Core techniques
│   └── methodology.md          # Implementation details
├── references/
│   └── README.md               # Key papers with links
└── src/
    ├── config.py               # IDM, MOBIL, simulation parameters
    ├── idm_jax.py              # IDM dynamics (JAX)
    ├── mobil_jax.py            # MOBIL incentive (JAX)
    ├── topology_jax.py         # Leader-finding, saltation vectors
    ├── simulator_jax.py        # Forward simulation + event recording
    ├── gradient_jax.py         # Backward pass (adjoint + saltation + ST-Gumbel)
    ├── inverse.py              # Adam optimizer
    ├── ablation.py             # Single-seed ablation
    ├── ablation_full.py        # Multi-seed ablation study
    └── animate.py              # Pygame visualization
```

## Current Status

🟡 **Week 1 of 4** — Experiments phase (targeting CDC 2026, deadline March 31)

### Done ✅
- Forward simulator: 20 vehicles, 2 lanes, bottleneck truck, IDM + MOBIL
- Analytical gradients: adjoint + saltation + ST-Gumbel backward pass
- Gradient validation: correct without lane changes (Mode 1)
- NaN handling: grazing guard, `nan_to_num`, gradient clipping
- Ablation study: 5 configurations designed and scripted
- Pygame time-space diagram visualization

### In Progress 🔧
- Running full ablation on NVIDIA A100 (5 configs × 5 seeds × 120 iterations)
- Performance bottleneck: backward pass at ~43s/iter due to Python-level JAX VJP loop

### Next Up 📋
- Analyze ablation results — does saltation + ST-Gumbel improve parameter recovery?
- Gradient validation against finite differences (all 5 configs)
- Refactor backward pass to `jax.lax.scan` for 10–50× speedup
- Generate publication-quality figures
- Write CDC paper (Sections I–IV)

## The Research Gap

Differentiable traffic simulation is a nearly empty niche (~5 papers). The closest work:

- **Son et al. (ICRA 2025):** DiffIDM — car-following only, no lane changes
- **Burger et al. (ITSC 2022):** MPCC for lane-change — ego-centric, not end-to-end differentiable
- **Contact mechanics (Dojo, Drake):** Has the right math but hasn't touched traffic

We bridge microscopic traffic modeling, hybrid systems theory, and differentiable programming.

## Citation

```
@inproceedings{diffidm_lc2026,
  title={Differentiating Through Lane-Change Events in Microscopic Traffic Simulation},
  author={TBD},
  booktitle={IEEE Conference on Decision and Control (CDC)},
  year={2026},
  note={Under preparation}
}
```

## License

TBD
