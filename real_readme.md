# diffIDM_LC — Differentiable Lane-Change for Microscopic Traffic Simulation

**A fully differentiable IDM+MOBIL traffic simulation pipeline for gradient-based calibration of car-following and lane-change parameters.**

## Motivation

Microscopic traffic simulators (IDM + MOBIL) are essential for modeling multi-lane traffic, but their discrete lane-change events break standard automatic differentiation. When a vehicle changes lanes, the leader-follower graph switches discontinuously — the dynamics operator itself changes. Naively autodiffing through this produces zero or biased gradients.

We build a fully differentiable pipeline by combining three components:

| Component | Handles | Technique |
|-----------|---------|-----------|
| Differentiable IDM | Smooth car-following dynamics | JAX reverse-mode AD |
| Sigmoid-relaxed MOBIL | Discrete lane-change decision | σ(h/τ) with temperature τ |
| Saltation matrix | Topology switch (state jump at LC) | Hybrid systems theory (Burden et al. 2016) |

**Key insight:** The sigmoid relaxation at small τ preserves nearly binary lane-change physics while providing smooth gradients everywhere. This outperforms Gumbel-Softmax, which fails when MOBIL decisions are decisive (incentive h ≫ 0, causing gradient saturation).

## Key Results

### IDM Parameter Estimation (from macroscopic observations)

Temperature sweep (τ = 0.1, 0.5, 1.0, 2.0), 20 vehicles, 2 lanes, 120 iterations:

| τ | Mean Error | v₀ | T | a | b | Lane Changes |
|---|-----------|-----|------|------|------|------|
| **0.1** | **11.1%** | 5.5% | 30.6% | 1.5% | 6.9% | 10 |
| 0.5 | 18.0% | 5.6% | 31.2% | 27.2% | 8.0% | 45 |
| 1.0 | 21.6% | 5.8% | 33.0% | 24.5% | 23.1% | 54 |
| 2.0 | 14.9% | 6.5% | 28.8% | 9.4% | 14.8% | 96 |

Baseline (hard MOBIL + saltation only): **30.6%** mean error → sigmoid relaxation reduces this to **11.1%**.

### MOBIL Parameter Reconstruction (from observed trajectories)

Estimating politeness p and threshold Δa_th with known heterogeneous IDM parameters:

| Config | Description | p err | Δa_th err | Mean | Gradient? |
|--------|-------------|-------|-----------|------|-----------|
| A | Full (logit + saltation + sig grad) | 8.6% | 21.6% | 15.1% | ✅ |
| B | No saltation | 5.4% | 22.5% | 13.9% | ✅ |
| C | No sigmoid grad | 60.0% | 100.0% | 80.0% | ❌ zero |
| D | Hard MOBIL | 60.0% | 100.0% | 80.0% | ❌ zero |
| E | Opposite init (adversarial) | 11.3% | 7.5% | **9.4%** | ✅ |

**Smoking gun:** Configs C and D produce **exactly zero gradient** for MOBIL parameters — proving that the sigmoid relaxation is essential for differentiability through lane-change decisions.

### Structural Findings

- **T non-identifiability:** Desired time headway T remains at ~30% error across all methods, due to structural coupling with b through the desired gap s* = vT + vΔv/(2√(ab)).
- **CF–LC decoupling:** IDM and MOBIL operate as parallel systems — in free flow, MOBIL is active but IDM params are invisible to LC decisions; in congestion, IDM dominates but MOBIL is suppressed. This is a model-intrinsic limitation, not a gradient quality issue.

## Quick Start

```bash
conda activate difflc_gpu   # or difflc (CPU)
cd src/
python3 run.py               # forward sim + gradient validation
python3 ablation.py          # ablation study (5 configs)
python3 experiment.py        # temperature sweep
python3 run_multilane_test.py  # multi-lane Gumbel test (3+ lanes)
```

## Documentation

- **[Problem Statement](doc/problem-statement.md)** — What we're solving and why it's hard
- **[ST-Gumbel & Saltation](doc/st-gumbel-and-saltation.md)** — Core gradient techniques explained
- **[Methodology](doc/methodology.md)** — Implementation details and experiment design

## Project Structure

```
diffIDM_LC/
├── real_readme.md               # This file (detailed documentation)
├── doc/
│   ├── problem-statement.md     # Problem formulation
│   ├── st-gumbel-and-saltation.md  # Core techniques
│   └── methodology.md          # Implementation details
├── references/
│   └── README.md               # Key papers with links
└── src/                        # → see diffLC repo for implementation
```

> **Note:** Source code lives in the [diffLC](https://github.com/zhengyouhan/diffLC) working repository. This repo serves as the documentation and presentation hub.

## Evolution of Approach

1. **Phase 1:** ST-Gumbel-Softmax + saltation matrices (Feb 2026)
   - Gumbel failed on decisive MOBIL decisions (gradient saturation)
2. **Phase 2:** Sigmoid relaxation (Logit-MOBIL) + saltation (Mar 2026)
   - σ(h/τ) with small τ: nearly hard switching + smooth gradients → **11.1% error**
3. **Current:** Structural diagnosis of MOBIL limitations + multi-lane Gumbel extension
   - 3-way Gumbel-Softmax works on 3+ lanes (decisions genuinely uncertain)
   - Leader-selection model proposed as unified CF+LC replacement (future work)

## The Research Gap

Differentiable traffic simulation is a nearly empty niche (~5 papers):

- **Son et al. (ICRA 2025):** DiffIDM — car-following only, no lane changes
- **Burger et al. (ITSC 2022):** MPCC for lane-change — ego-centric, not end-to-end differentiable
- **Contact mechanics (Dojo, Drake):** Has the right math but hasn't touched traffic

We bridge microscopic traffic modeling, hybrid systems theory, and differentiable programming.

## References

- Treiber, Hennecke, Helbing (2000) — Intelligent Driver Model
- Kesting, Treiber, Helbing (2007) — MOBIL lane-change model
- Son et al. (2024) — DiffIDM: differentiable IDM (ICRA 2025)
- Burden et al. (2016) — Saltation matrices for piecewise-differentiable flows
- Jang, Gu, Poole (2017) — Gumbel-Softmax categorical reparameterization
- Vilar & Saiz (2026) — Differentiable discrete stochastic simulation

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
