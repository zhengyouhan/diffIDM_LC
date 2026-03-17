# diffIDM_LC — Differentiable Multi-Lane Traffic Simulation via Continuous Lateral Dynamics

**A fully differentiable IDM+MOBIL traffic simulation for gradient-based calibration of car-following and lane-change parameters. No discrete events, no saltation matrices — one smooth ODE.**

## Motivation

Microscopic traffic simulators (IDM + MOBIL) are essential for modeling multi-lane traffic, but their discrete lane-change events break standard automatic differentiation. When a vehicle changes lanes, the leader-follower graph switches discontinuously — the dynamics operator itself changes. Naively autodiffing through this produces zero or biased gradients.

**Our approach:** replace discrete lane-change events with **continuous lateral dynamics**. Each vehicle has a physical lateral position $y_i(t)$ that evolves as a smooth ODE. Lane membership, leader selection, and inter-vehicle interactions are all determined by physical proximity — not by discrete assignments or relaxed probabilities.

The entire traffic model is **one coupled ODE system** $(\dot{x}, \dot{v}, \dot{y})$, differentiable end-to-end via standard reverse-mode AD. No saltation matrices, no Gumbel-Softmax, no event detection, no graph rewiring.

Lane changes **emerge** from physical lateral movement, not binary switches.

## Key Results

### Gradient Quality (Phase 3 — Continuous Lateral Dynamics)

Autodiff vs. finite differences, 8 vehicles, 3 lanes, 5 active lane changes:

| Parameter | Autodiff | Finite Diff | Relative Error |
|---|---|---|---|
| $v_0$ | -10.544525 | -10.544525 | 2.5 × 10⁻⁹ |
| $T$ | 498.136819 | 498.136824 | 1.1 × 10⁻⁸ |
| $s_0$ | 33.394424 | 33.394424 | 8.8 × 10⁻¹⁰ |
| $a$ | 340.748438 | 340.748438 | 1.9 × 10⁻⁹ |
| $b$ | -248.356289 | -248.356289 | 1.5 × 10⁻⁹ |

All parameters match to **8–10 significant digits** with active lane changes in the scenario.

### IDM Parameter Estimation (Phase 2 — Sigmoid MOBIL, historical)

Temperature sweep (τ = 0.1, 0.5, 1.0, 2.0), 20 vehicles, 2 lanes, 120 iterations:

| τ | Mean Error | v₀ | T | a | b | Lane Changes |
|---|-----------|-----|------|------|------|------|
| **0.1** | **11.1%** | 5.5% | 30.6% | 1.5% | 6.9% | 10 |
| 0.5 | 18.0% | 5.6% | 31.2% | 27.2% | 8.0% | 45 |
| 1.0 | 21.6% | 5.8% | 33.0% | 24.5% | 23.1% | 54 |
| 2.0 | 14.9% | 6.5% | 28.8% | 9.4% | 14.8% | 96 |

Baseline (hard MOBIL + saltation only): **30.6%** → sigmoid relaxation: **11.1%**.

### MOBIL Parameter Reconstruction Ablation (Phase 2, historical)

| Config | Description | p err | Δa_th err | Mean | Gradient? |
|--------|-------------|-------|-----------|------|-----------|
| A | Full (logit + saltation + sig grad) | 8.6% | 21.6% | 15.1% | ✅ |
| B | No saltation | 5.4% | 22.5% | 13.9% | ✅ |
| C | No sigmoid grad | 60.0% | 100.0% | 80.0% | ❌ zero |
| D | Hard MOBIL | 60.0% | 100.0% | 80.0% | ❌ zero |
| E | Opposite init (adversarial) | 11.3% | 7.5% | **9.4%** | ✅ |

**Smoking gun:** Configs C and D produce **exactly zero gradient** for MOBIL parameters — proving that decision relaxation is essential for differentiability through lane-change decisions.

### Structural Findings

- **T non-identifiability:** Desired time headway $T$ remains at ~30% error across all methods, due to structural coupling with $b$ through $s^* = vT + v\Delta v / (2\sqrt{ab})$.
- **CF–LC decoupling:** IDM and MOBIL operate as parallel systems with mutual suppression. This is a model-intrinsic limitation, not a gradient quality issue.
- **Mid-lane equilibrium:** Gaussian lane weights create blended headways at mid-lane that can be better than either pure lane. Committed LC perception (ego follows target lane leader) eliminates this.

## Approach Evolution

| Phase | Approach | Key Result | Status |
|-------|----------|------------|--------|
| Phase 1 | ST-Gumbel-Softmax + saltation | Gumbel saturates on decisive MOBIL decisions | Abandoned |
| Phase 2 | Sigmoid relaxation (Logit-MOBIL) + saltation | 11.1% error, proves sigmoid essential | Validated |
| **Phase 3** | **Continuous lateral dynamics** | **Exact gradients (10⁻⁹), no discrete machinery** | **Current (CDC)** |
| Future | AV convex Φ planning layer | Φ-optimization natural for autonomous vehicles | Planned |

**Key lesson:** Phase 2 proved that sigmoid relaxation gives good gradients through lane-change decisions. Phase 3 goes further — by making lane changes physically continuous, we eliminate the need for any relaxation or hybrid-systems machinery entirely.

## Quick Start

```bash
conda activate difflc_gpu   # or difflc (CPU)
cd src/
python3 lateral_3lane_mvp.py    # Phase 3: continuous lateral dynamics (current)
python3 run.py                  # Phase 2: sigmoid MOBIL + saltation
python3 ablation.py             # Phase 2: ablation study (5 configs)
python3 experiment.py           # Phase 2: temperature sweep
```

## Documentation

- **[Problem Statement](doc/problem-statement.md)** — What we're solving and why it's hard
- **[Methodology](doc/methodology.md)** — Current formulation: continuous lateral dynamics
- **[Gradient Techniques](doc/gradient-techniques.md)** — Evolution of gradient methods (Phase 1→3)

## Project Structure

```
diffIDM_LC/
├── real_readme.md               # This file (detailed documentation)
├── doc/
│   ├── problem-statement.md     # Problem formulation
│   ├── methodology.md           # Current approach (continuous lateral dynamics)
│   └── gradient-techniques.md   # Gradient methods across all phases
├── references/
│   └── README.md               # Key papers with links
└── src/                        # → see diffLC repo for implementation
```

> **Note:** Source code lives in the [diffLC](https://github.com/zhengyouhan/diffLC) working repository. This repo serves as the documentation and presentation hub.

## The Research Gap

Differentiable traffic simulation is a nearly empty niche (~5 papers):

- **Son et al. (ICRA 2025):** DiffIDM — car-following only, no lane changes
- **Burger et al. (ITSC 2022):** MPCC for lane-change — ego-centric, not end-to-end differentiable
- **Andelfinger (2021):** Differentiable ABM — concept, not vehicular traffic calibration
- **Contact mechanics (Dojo, Drake):** Has the right math but hasn't touched traffic

Nobody does differentiable simulation for multi-lane traffic calibration or inverse problems.

## Two-Paper Strategy

- **Paper A (CDC 2026):** Human-driver 3-lane lateral dynamics + MOBIL softmax. Due March 31.
- **Paper B (future):** AV convex Φ planning layer. Shared machinery (soft-min headway, lateral dynamics).

## References

- Treiber, Hennecke, Helbing (2000) — Intelligent Driver Model
- Kesting, Treiber, Helbing (2007) — MOBIL lane-change model
- Son et al. (2024) — DiffIDM: differentiable IDM (ICRA 2025)
- Burden et al. (2016) — Saltation matrices for piecewise-differentiable flows
- Jang, Gu, Poole (2017) — Gumbel-Softmax categorical reparameterization
- Li et al. (2020) — Incremental Potential Contact (IPC)
- Wang et al. (2024) — APF for mandatory lane changes (TGPF model)
- Le Cleac'h et al. (2023) — Dojo: differentiable contact dynamics

## Citation

```
@inproceedings{diffidm_lc2026,
  title={Differentiable Multi-Lane Traffic Simulation via Continuous Lateral Dynamics},
  author={TBD},
  booktitle={IEEE Conference on Decision and Control (CDC)},
  year={2026},
  note={Under preparation}
}
```
