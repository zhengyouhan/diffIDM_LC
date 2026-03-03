# Current Methodology

## Overview

We implement a **composed backward pass** through a microscopic traffic simulator with lane changes. The approach treats the simulator as a hybrid dynamical system and combines three gradient components:

1. **Adjoint method** through IDM dynamics (smooth arcs)
2. **Saltation matrices** at topology switches (dynamics jump correction)
3. **Straight-Through Gumbel** through discrete lane-change decisions

We call this **Strategy 1.5** — it preserves the exact discrete simulation in the forward pass while providing corrected gradients in the backward pass.

→ See [st-gumbel-and-saltation.md](st-gumbel-and-saltation.md) for detailed exposition of components (2) and (3).

---

## 1. Forward Model

### 1.1 State Representation

$N$ vehicles on a 2-lane road. Continuous state:

$$\mathbf{z} = (x_1, v_1, x_2, v_2, \ldots, x_N, v_N) \in \mathbb{R}^{2N}$$

Discrete state: lane assignments $\boldsymbol\ell = (\ell_1, \ldots, \ell_N) \in \{0,1\}^N$.

### 1.2 IDM Car-Following

Under interaction graph $G(\boldsymbol\ell)$, each vehicle follows the IDM:

$$\dot{v}_i = a\left[1 - \left(\frac{v_i}{v_0}\right)^4 - \left(\frac{s^*(v_i, \Delta v_i)}{s_i}\right)^2\right]$$

$$s^*(v, \Delta v) = \text{softplus}\left(s_0 + vT + \frac{v\,\Delta v}{2\sqrt{ab}}\right)$$

Parameters to estimate: $\theta = (v_0, T, a, b)$. We fix $s_0 = 2.0$ and $\delta = 4$.

Euler integration: $\mathbf{z}^{k+1} = \mathbf{z}^k + \Delta t \, F_G(\mathbf{z}^k; \theta)$ with $\Delta t = 0.1$s.

### 1.3 MOBIL Lane-Changing

Every $\Delta t_{\text{LC}} = 1.0$s, each vehicle evaluates the MOBIL incentive $h_i(\mathbf{z};\theta)$. If $h_i > 0$ and the safety constraint holds ($a_{\text{new follower}} \geq -b_{\text{safe}}$), the lane change executes:

1. Lane assignment flips: $\ell_i \gets 1 - \ell_i$
2. Leader-follower graph updates: $G \to G'$
3. Dynamics operator changes: $F_G \to F_{G'}$
4. Event recorded with saltation info for backward pass

A cooldown of 3.0s prevents rapid re-switching.

### 1.4 Scenario

- 20 vehicles, 2 lanes, 1000m road
- Vehicle 0 is a slow truck ($v_0^{\text{truck}} = 18$ m/s) creating a bottleneck
- Simulation: 8s warmup (fixed params) + 60s main (parameterized)
- Warmup uses default $\theta$ so initial state is $\theta$-independent

---

## 2. Backward Pass

### 2.1 Cost Function

Position MSE with optional velocity matching:

$$J(\theta) = \frac{1}{NK}\sum_{i,k}\left[\left(x_i^{\text{sim}} - x_i^{\text{obs}}\right)^2 + \beta\left(v_i^{\text{sim}} - v_i^{\text{obs}}\right)^2\right]$$

Velocity observations ($\beta = 1.0$) significantly improve identifiability of $T$ and $b$.

### 2.2 Adjoint Through Euler Steps

Between lane-change events, the adjoint propagates via JAX's reverse-mode AD:

$$(\lambda_\mathbf{z}, \lambda_\theta) = \text{vjp}(\Phi_G, \mathbf{z}^k, \theta)^\top(\lambda)$$

$$\text{grad}_\theta \mathrel{+}= \lambda_\theta, \qquad \lambda \gets \lambda_\mathbf{z} + \partial J_k / \partial\mathbf{z}^k$$

### 2.3 Saltation Correction

At each recorded lane-change event $j$:

$$\lambda \gets \lambda + \mathbf{w}_j\,(\mathbf{u}_j^\top \lambda)$$

where $\mathbf{u}_j = \Delta F_j / (D_\mathbf{z} h_j \cdot F_j^-)$ and $\mathbf{w}_j = D_\mathbf{z} h_j$.

Grazing guard: $|D_\mathbf{z} h \cdot F^-| \geq 0.01$.

### 2.4 ST-Gumbel Decision Gradient

At each event $j$:

$$\gamma_j = \frac{1}{\tau}\,y_{j,1}^{\text{soft}}(1 - y_{j,1}^{\text{soft}}) \cdot \lambda^\top \Delta F_j \cdot \Delta t$$

$$\text{grad}_\theta \mathrel{+}= \gamma_j \cdot \frac{\partial h_j}{\partial\theta}$$

### 2.5 Post-Processing

1. **NaN safety:** `nan_to_num(grad, nan=0, posinf=1e3, neginf=-1e3)`
2. **Gradient clipping:** Global norm clipped to 10.0
3. **Tikhonov regularization:** $J \mathrel{+}= \alpha\|\theta - \theta_{\text{init}}\|^2$ with $\alpha = 0.01$

---

## 3. Optimization

- **Optimizer:** Adam (lr=0.02, manual implementation)
- **Parameter clipping:** $v_0 \in [15, 50]$, $T \in [0.5, 3.0]$, $a \in [0.5, 3.0]$, $b \in [0.5, 5.0]$
- **Initial guess:** 15% perturbation from true values
- **Iterations:** 120 per run

---

## 4. Ablation Study Design

Five configurations to isolate each component's contribution:

| Config | Saltation | ST-Gumbel | Lane Changes | Tests |
|--------|-----------|-----------|-------------|-------|
| **E) Baseline** | ✗ | ✗ | Disabled | IDM-only reference |
| **A) Plain adjoint** | ✗ | ✗ | Enabled | Naive autodiff (ignores LC) |
| **B) Saltation only** | ✓ | ✗ | Enabled | Topology correction only |
| **C) ST-Gumbel only** | ✗ | ✓ | Enabled | Decision gradient only |
| **D) Full Strategy 1.5** | ✓ | ✓ | Enabled | Complete method |

Each configuration runs with 5 random seeds for statistical robustness. Metrics: final cost, parameter error (%), convergence speed.

**Expected outcome:** Config D (full) should outperform A (naive) most clearly when lane changes are frequent. Config E (baseline) validates the IDM gradient machinery.

---

## 5. Implementation Stack

- **JAX** — automatic differentiation + GPU acceleration
- **Python 3.12** — simulation loop (currently Python-level; `jax.lax.scan` refactor planned)
- **Hardware:** NVIDIA A100 80GB (Azure)

### Key Files

```
src/
├── config.py          — IDM, MOBIL, simulation parameters
├── idm_jax.py         — IDM dynamics + Euler step (JAX)
├── mobil_jax.py        — MOBIL incentive + gradients (JAX)
├── topology_jax.py     — Leader finding, saltation vectors
├── simulator_jax.py    — Forward simulation with event recording
├── gradient_jax.py     — Composed backward pass (adjoint + saltation + ST-Gumbel)
├── inverse.py          — Adam optimizer, parameter clipping
├── ablation.py         — Single ablation run
└── ablation_full.py    — Full ablation: 5 configs × 5 seeds
```

---

## 6. Known Limitations & Next Steps

### Current Limitations
- **Speed:** Backward pass uses a Python for-loop over 600 timesteps, each calling `jax.vjp`. This limits throughput to ~51s/iteration despite A100 GPU. A `jax.lax.scan` refactor could yield 10–50× speedup.
- **Identifiability:** Parameters $T$ and $b$ are structurally correlated in IDM — difficult to recover independently from position-only data.
- **Gradient clipping:** All gradient norms saturate at the clip ceiling (10.0), losing magnitude information.
- **Scenario scale:** 20 vehicles, synthetic data. Real-world validation pending.

### Planned Improvements
1. Refactor backward pass to `jax.lax.scan` for GPU-native execution
2. Investigate population-level NCP formulation (inspired by Dojo's single-level approach)
3. Gradient validation against finite differences for all 5 configurations
4. Extend to heterogeneous fleets (per-vehicle parameters)
