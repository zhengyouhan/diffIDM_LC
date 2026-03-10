# Methodology

## Overview

We implement a **fully differentiable microscopic multi-lane traffic simulation** that combines differentiable IDM car-following, sigmoid-relaxed MOBIL lane-changing, and saltation matrix corrections at topology switches. The pipeline enables gradient-based parameter estimation via backpropagation through the full simulation trajectory, including lane-change events.

→ See [st-gumbel-and-saltation.md](st-gumbel-and-saltation.md) for detailed exposition of each gradient component.

---

## 1. Forward Model

### 1.1 State Representation

$N$ vehicles on a multi-lane road. Continuous state:

$$\mathbf{z} = (x_1, v_1, x_2, v_2, \ldots, x_N, v_N) \in \mathbb{R}^{2N}$$

Discrete state: lane assignments $\boldsymbol\ell = (\ell_1, \ldots, \ell_N) \in \{0,1,\ldots,L-1\}^N$.

### 1.2 IDM Car-Following

Under interaction graph $G(\boldsymbol\ell)$, each vehicle follows the IDM:

$$\dot{v}_i = a\left[1 - \left(\frac{v_i}{v_0}\right)^4 - \left(\frac{s^*(v_i, \Delta v_i)}{s_i}\right)^2\right]$$

$$s^*(v, \Delta v) = \text{softplus}\left(s_0 + vT + \frac{v\,\Delta v}{2\sqrt{ab}}\right)$$

Parameters: $\theta_{\text{CF}} = (v_0, T, a, b)$. Fixed: $s_0 = 2.0$, $\delta = 4$.

Euler integration: $\mathbf{z}^{k+1} = \mathbf{z}^k + \Delta t \, F_G(\mathbf{z}^k; \theta)$ with $\Delta t = 0.1$s.

### 1.3 MOBIL Lane-Changing with Sigmoid Relaxation

Every $\Delta t_{\text{LC}} = 1.0$s, each vehicle evaluates the MOBIL incentive $h_i(\mathbf{z};\theta)$.

**Hard MOBIL:** $a_i = \mathbb{1}[h_i > 0]$ — not differentiable.

**Logit-MOBIL:** $P(\text{LC}_i) = \sigma(h_i / \tau)$ — differentiable, with temperature $\tau$ controlling sharpness.

The incentive function:
$$h_i = (\tilde{a}_i - a_i) + p(\Delta a_{\text{neighbors}}) - \Delta a_{\text{th}}$$

Parameters: $\theta_{\text{LC}} = (p, \Delta a_{\text{th}})$.

Safety constraint: $a_{\text{new follower}} \geq -b_{\text{safe}}$. Cooldown of 3.0s prevents rapid re-switching.

### 1.4 Multi-Lane Extension (3+ lanes)

For 3+ lanes, the binary decision becomes a multi-way choice. We use Gumbel-Softmax over logits:

$$\mathbf{p} = \text{softmax}\left(\frac{(h_{\text{left}},\; 0,\; h_{\text{right}}) + \mathbf{g}}{\tau}\right), \quad \mathbf{g} \sim \text{Gumbel}(0,1)$$

This works because with 3 options, at least one pair of probabilities is genuinely uncertain, preventing gradient saturation.

---

## 2. Backward Pass

### 2.1 Cost Functions

**IDM estimation (macro loss):** Simulated trajectories → Gaussian kernel smoothing → macroscopic fields (ρ, q) → MSE vs. sensor data.

**MOBIL reconstruction (micro loss):** Direct trajectory MSE:
$$J(\theta) = \frac{1}{NK}\sum_{i,k}\left(x_i^{\text{sim}} - x_i^{\text{obs}}\right)^2$$

### 2.2 Composed Backward Sweep

Three components compose in the backward sweep (see [st-gumbel-and-saltation.md](st-gumbel-and-saltation.md)):

1. **Adjoint through Euler steps** — JAX reverse-mode AD
2. **Sigmoid decision gradient** — $\sigma'(h/\tau) \cdot \lambda^\top \Delta F \cdot \Delta t$
3. **Saltation correction** — $\lambda \gets \Xi^\top \lambda$ at each LC event

### 2.3 Post-Processing

- **NaN safety:** `nan_to_num(grad, nan=0, posinf=1e3, neginf=-1e3)`
- **Gradient clipping:** Global norm clipped to 10.0

---

## 3. Optimization

- **Optimizer:** Adam (lr=0.02)
- **Parameter clipping:** $v_0 \in [15, 50]$, $T \in [0.5, 3.0]$, $a \in [0.5, 3.0]$, $b \in [0.5, 5.0]$
- **Iterations:** 120 per run

---

## 4. Experiments and Results

### 4.1 Temperature Sweep (IDM estimation)

20 vehicles, 2 lanes, 120 iterations, macroscopic loss:

| τ | Mean Error | v₀ | T | a | b | Lane Changes |
|---|-----------|-----|------|------|------|------|
| **0.1** | **11.1%** | 5.5% | 30.6% | 1.5% | 6.9% | 10 |
| 0.5 | 18.0% | 5.6% | 31.2% | 27.2% | 8.0% | 45 |
| 1.0 | 21.6% | 5.8% | 33.0% | 24.5% | 23.1% | 54 |
| 2.0 | 14.9% | 6.5% | 28.8% | 9.4% | 14.8% | 96 |

Baseline (hard MOBIL + saltation): **30.6%** → sigmoid relaxation: **11.1%**.

Trade-off: larger τ produces more spurious lane changes, distorting the physics. At τ=0.1, lane changes remain nearly binary.

### 4.2 MOBIL Reconstruction Ablation

Recovering MOBIL params (p, Δa_th) from observed trajectories, known heterogeneous IDM:

| Config | Description | p err | Δa_th err | Mean | Gradient? |
|--------|-------------|-------|-----------|------|-----------|
| A | Full (logit + salt + sig) | 8.6% | 21.6% | 15.1% | ✅ |
| B | No saltation | 5.4% | 22.5% | 13.9% | ✅ |
| C | No sigmoid grad | 60.0% | 100.0% | 80.0% | ❌ zero |
| D | Hard MOBIL | 60.0% | 100.0% | 80.0% | ❌ zero |
| E | Opposite init | 11.3% | 7.5% | **9.4%** | ✅ |

**Key findings:**
- **Configs C, D:** Zero gradient for MOBIL parameters — proves sigmoid relaxation is essential
- **A ≈ B:** Saltation negligible in sparse-LC regime (1-2 events). Expected to matter more in congested scenarios with abundant lane changes.
- **Config E:** Converges from adversarial initialization (opposite direction), achieving best mean error

### 4.3 Moderate-Density Scenario

12 vehicles, 83m spacing:

| Config | Mean Error | a | b |
|--------|-----------|------|------|
| Plain adjoint | 33.7% | 64.3% | 36.4% |
| + Saltation | 30.7% | 64.3% | 24.4% |

Saltation improves `b` recovery (36.4% → 24.4%). Parameter `a` hits lower bound in all configs.

---

## 5. Structural Findings

### 5.1 T Non-Identifiability
The desired time headway T remains at ~30% error across **all** methods and scenarios. This is structural: T and b are coupled through the desired gap:

$$s^* = vT + \frac{v\Delta v}{2\sqrt{ab}}$$

T and b create a correlated valley in the loss landscape — the optimizer converges to ~34% T error from both directions.

### 5.2 CF–LC Decoupling
IDM and MOBIL operate as parallel systems with mutual suppression:
- **Free flow:** MOBIL is active, IDM is passive → IDM parameters are invisible to lane-change decisions
- **Congestion:** IDM dominates, MOBIL is suppressed → lane changes add no information

This is a **model-intrinsic limitation**, not a gradient quality issue. No amount of gradient engineering can overcome structural decoupling.

### 5.3 Implication
MOBIL was designed for forward simulation (2007), never for inversion. The structural decoupling suggests that a unified model — where car-following and lane-changing share a single objective function — would be fundamentally better suited for inverse problems. See the leader-selection model concept in [ideas](../idea/).

---

## 6. Implementation

- **JAX** — automatic differentiation + GPU acceleration
- **Python 3.12** — simulation loop
- **Hardware:** NVIDIA A100 (Azure), RTX 4090 (local)

### Performance
- CPU (dt=0.1, T=60s, 120 iter): ~125 min → 11.1% error
- GPU (dt=0.2, T=30s, 60 iter): ~49 min → 18.3% error
- Bottleneck: Python-level VJP loop (~50s/iter). `jax.lax.scan` refactor planned.

---

## 7. Open Questions

1. Does saltation matter in congested flows with 10+ LC events per observation?
2. Can heterogeneous MOBIL (per-vehicle p, Δa_th) be reconstructed?
3. Does the 3-lane Gumbel-Softmax genuinely outperform sigmoid for multi-lane?
4. Can a two-stage approach (IDM first, then MOBIL) improve joint estimation?
