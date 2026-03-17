# Methodology — Continuous Lateral Dynamics

## Overview

We implement a **fully differentiable microscopic multi-lane traffic simulation** where lane changes are continuous physical movements, not discrete events. The model is a single smooth ODE system $(\dot{x}, \dot{v}, \dot{y})$ that can be differentiated end-to-end via standard reverse-mode AD.

No saltation matrices. No Gumbel-Softmax. No event detection. No graph rewiring.

→ See [gradient-techniques.md](gradient-techniques.md) for detailed comparison with prior discrete-event approaches.

---

## 1. Vehicle State

$N$ vehicles on a $K$-lane road. Each vehicle has continuous state:

$$\mathbf{z}_i = (x_i, v_i, y_i)$$

- $x_i$: longitudinal position [m]
- $v_i$: longitudinal velocity [m/s]
- $y_i$: lateral position [m]

Lane centers at $y_k$ for $k = 1, \ldots, K$ (e.g., $y_1 = 0$, $y_2 = 3.7$, $y_3 = 7.4$ m for 3 lanes with width $W = 3.7$ m).

Full state vector: $\mathbf{z} = (x_1, v_1, y_1, \ldots, x_N, v_N, y_N) \in \mathbb{R}^{3N}$.

---

## 2. Lane Weights from Physical Position

Vehicle $i$'s membership in each lane is determined by its physical lateral position:

$$w_i^{(k)} = \frac{\exp\left(-\gamma(y_i - y_k)^2\right)}{\sum_{k'} \exp\left(-\gamma(y_i - y_{k'})^2\right)}$$

where $\gamma > 0$ controls the sharpness of lane boundaries.

**Properties:**
- Vehicle at lane center: $w \approx (0, 1, 0)$ (in lane 2)
- Vehicle mid-transition: $w \approx (0.3, 0.5, 0.2)$
- Smooth and $C^\infty$ in $y_i$
- Weights come from physical position (a state variable with inertia), not from a relaxation or optimization decision — the car is literally at that position

---

## 3. Soft-Min Headway

For vehicle $i$ in lane $k$, the effective headway to the nearest leader:

$$\bar{s}_i^{(k)} = -\frac{1}{\mu} \ln \sum_{j:\, x_j > x_i} w_j^{(k)} \cdot e^{-\mu(x_j - x_i - L)}$$

where:
- $\mu > 0$: soft-min sharpness (receptive field $\sim 1/\mu$ meters)
- $w_j^{(k)}$: vehicle $j$'s presence in lane $k$ (from its lateral position)
- $L$: vehicle length

**Interpretation:** "What is the effective gap to the nearest vehicle ahead of me in lane $k$, where both 'nearest' and 'in lane $k$' are soft?"

Similarly, the soft velocity difference:

$$\Delta \bar{v}_i^{(k)} = v_i - \frac{\sum_{j:\, x_j > x_i} w_j^{(k)} \cdot e^{-\mu(x_j - x_i - L)} \cdot v_j}{\sum_{j:\, x_j > x_i} w_j^{(k)} \cdot e^{-\mu(x_j - x_i - L)} + \epsilon}$$

**Empty lane guard:** When no vehicle is ahead (sum → 0), $\bar{s} \to +\infty$ → free-flow acceleration.

**Gradient properties:**
- $\partial \bar{s}_i^{(k)} / \partial w_j^{(k)}$ decays as $e^{-\mu(x_j - x_i)}$ — automatic attention mechanism
- Nearby vehicles dominate; distant vehicles contribute exponentially small gradients

---

## 4. IDM with Blended Headway

Vehicle $i$'s acceleration is IDM, blended across lanes by its own lane weights:

$$a_i = \sum_k w_i^{(k)} \cdot a_i^{\text{IDM}}\left(v_i, \bar{s}_i^{(k)}, \Delta\bar{v}_i^{(k)};\, \theta\right)$$

where:

$$a_i^{\text{IDM}}(v, s, \Delta v;\, \theta) = a\left[1 - \left(\frac{v}{v_0}\right)^\delta - \left(\frac{s^*(v, \Delta v)}{s}\right)^2\right]$$

$$s^*(v, \Delta v) = \text{softplus}(s_0 + vT + v\Delta v / 2\sqrt{ab})$$

Parameters: $\theta_{\text{CF}} = (v_0, T, a, b)$. Fixed: $s_0 = 2.0$, $\delta = 4$.

**Interpretation:** A vehicle centered in lane 2 follows only lane 2's leader. A vehicle mid-transition follows a blend of adjacent lane leaders.

### Committed Perception (Bernie's Fix)

A subtlety: with pure Gaussian blending, a vehicle at mid-lane sees diluted leaders from both lanes → inflated headways → mid-lane equilibrium (free acceleration better than either lane).

**Fix:** Decouple ego perception from others' perception:
- **Ego headway:** Based on target lane (committed decision, near-binary softmax with $\tau_{\text{decide}} = 0.05$)
- **Others' headway:** Based on physical $y$ position (Gaussian blocking — the vehicle physically occupies space in both lanes)

This eliminates the mid-lane equilibrium while preserving physical blocking behavior for surrounding vehicles.

---

## 5. 3-Way MOBIL Softmax Decision

The lane-change decision is a softmax over MOBIL incentives for all $K$ lanes.

### Per-Lane Incentive

$$I_i^{(k)} = a_i^{(k)} - \bar{a}_i + \text{politeness}_i^{(k)}$$

where:
- $a_i^{(k)}$ = IDM acceleration in lane $k$
- $\bar{a}_i$ = current blended acceleration
- $\text{politeness}_i^{(k)}$ = continuous politeness term (§6)

### Adjusted Incentives

The "stay" option (current lane) is the baseline:

$$\tilde{I}_i^{(k)} = \begin{cases} 0 & \text{if } k = k_i^{\text{current}} \\ I_i^{(k)} - \Delta a_{\text{th}} & \text{otherwise} \end{cases}$$

where $k_i^{\text{current}} = \arg\min_k |y_i - y_k|$.

### Softmax Target

$$\text{prob}_i^{(k)} = \frac{\exp(\tilde{I}_i^{(k)} / \tau)}{\sum_{k'} \exp(\tilde{I}_i^{(k')} / \tau)}$$

$$y_i^{\text{target}} = \sum_k \text{prob}_i^{(k)} \cdot y_k$$

The temperature $\tau$ controls **decision sharpness only**, not physical lateral speed. These are decoupled:

| Parameter | Controls | Affects |
|---|---|---|
| $\tau$ | Decision sharpness in softmax | Gradient signal strength |
| $\kappa$ | Lateral responsiveness | How quickly vehicle starts moving |
| $u_{\max}$ | Maximum lateral speed | Lane-change duration |

---

## 6. Continuous Politeness

### The Problem with Discrete Politeness

Standard MOBIL computes politeness as the acceleration change for discrete "new follower" and "old follower" before/after the lane change. With continuous lateral dynamics, there is no discrete switch — vehicle $i$ gradually slides between lanes, continuously affecting all nearby vehicles.

### Continuous Formulation

$$\text{politeness}_i^{(k)} = p \cdot \frac{\partial}{\partial w_i^{(k)}} \sum_{j \neq i} a_j$$

This is the **marginal impact** of vehicle $i$'s lane-$k$ presence on all other vehicles' accelerations.

**Interpretation:**
- **Entering lane $k$:** followers in lane $k$ see a closer leader → their headway shrinks → acceleration drops. The derivative captures this.
- **Leaving current lane:** followers behind lose a leader → headway increases → acceleration improves.
- **Distance weighting:** The gradient decays exponentially with distance — no need to explicitly identify "new follower" or "old follower."

### Relationship to Discrete MOBIL

In the limit where lane weights are binary and only the nearest follower has non-negligible gradient, the continuous politeness reduces to the standard MOBIL term $p \cdot [(\tilde{a}_n - a_n) + (\tilde{a}_o - a_o)]$. The continuous version is strictly more general.

### Implementation

```python
def total_others_acc(w_all, x, v, theta, mu, i):
    accs = blended_acc_all(x, v, w_all, theta, mu)
    return jnp.sum(accs) - accs[i]

grad_w = jax.grad(total_others_acc)(w_all, x, v, theta, mu, i)
politeness_ik = p * grad_w[i, k]
```

One backward pass per vehicle gives politeness for all $K$ lanes simultaneously.

---

## 7. Lateral Dynamics

$$\dot{y}_i = u_{\max} \cdot \tanh\left(\frac{\kappa(y_i^{\text{target}} - y_i)}{u_{\max}}\right)$$

where:
- $y_i^{\text{target}}$: softmax-weighted lane center (from §5)
- $\kappa > 0$: proportional gain
- $u_{\max} \approx 1.2$ m/s: maximum lateral velocity (typical for 3–5s lane changes)

**Why tanh, not clip:** `clip` has zero gradient at saturation. In test scenarios, 61% of vehicle-timesteps hit saturation — that's 61% dead lateral gradients. $\tanh$ gives the same asymptotic behavior ($\dot{y} \to \pm u_{\max}$) but the gradient $\text{sech}^2(\cdot)$ is always nonzero.

### Physical Constraints

- Road boundaries: $y_i \in [0, (K-1) \cdot W]$
- Non-negative velocity: $v_i \geq 0$

---

## 8. Full System ODE

$$\dot{x}_i = v_i$$

$$\dot{v}_i = a_i(x, v, y;\, \theta) \quad \text{(IDM with blended headway, §4)}$$

$$\dot{y}_i = u_{\max} \cdot \tanh\left(\frac{\kappa(y_i^{\text{target}}(x, v, y;\, \theta) - y_i)}{u_{\max}}\right)$$

for $i = 1, \ldots, N$.

**Integration:** Euler or RK4. $\Delta t = 0.1$s.

**No discrete events.** Lane changes happen when $\dot{y}_i \neq 0$, which is whenever the MOBIL softmax prefers a different lane. The vehicle physically moves, lane weights shift, and interactions evolve smoothly.

---

## 9. Differentiability

### Every Component Is Smooth

| Component | Smoothness |
|---|---|
| Lane weights $w_i^{(k)}(y_i)$ | Softmax of Gaussian — $C^\infty$ |
| Soft-min headway $\bar{s}_i^{(k)}$ | Log-sum-exp — $C^\infty$ |
| IDM acceleration | $C^\infty$ (softplus for $s^*$) |
| MOBIL softmax | $C^\infty$ |
| Continuous politeness | Gradient of smooth functions — $C^\infty$ |
| Lateral velocity (tanh) | $C^\infty$ |

### Gradient Method

Standard reverse-mode AD (JAX `jax.grad`) through unrolled simulation. No custom adjoint needed — JAX handles the chain rule through Euler steps automatically.

For long simulations, the continuous adjoint method reduces memory. But for the current scale (50–100 steps), unrolled autodiff is sufficient.

---

## 10. Hyperparameters

| Symbol | Name | Role | Range | Notes |
|---|---|---|---|---|
| $\gamma$ | Lane weight sharpness | Lane boundary crispness | 1.0–5.0 m⁻² | Higher = sharper |
| $\mu$ | Soft-min sharpness | Leader selection range | 0.05–0.5 m⁻¹ | $1/\mu$ = effective range |
| $\tau$ | MOBIL softmax temperature | Decision gradient signal | 0.1–1.0 m/s² | Annealable |
| $\kappa$ | Lateral proportional gain | LC responsiveness | 0.5–3.0 s⁻¹ | Avoid extreme saturation |
| $u_{\max}$ | Max lateral velocity | LC duration | 0.8–1.5 m/s | ~3–5s for full LC |
| $p$ | Politeness factor | Cooperation level | 0.0–0.5 | Standard MOBIL range |
| $\Delta a_{\text{th}}$ | LC threshold | Incentive required | 0.1–0.5 m/s² | Standard MOBIL range |

---

## 11. Open Challenges

### 11.1 Mid-Lane Equilibrium

Gaussian lane weights create blended headways at mid-lane that can be better than either pure lane. The committed perception fix (§4) addresses this but introduces a near-hard decision boundary via $\tau_{\text{decide}}$.

### 11.2 LC Oscillation

Without cooldown, vehicles can rapidly switch preferred lanes (softmax oscillates between left/stay/right). Treiber's original MOBIL uses a cooldown timer — the continuous analog is under exploration (damping, hysteresis, frustration gating).

### 11.3 Gradient Explosion at Long Horizons

Autodiff through 1200+ chaotic timesteps causes Jacobian chain explosion. Gradient checkpointing or truncated BPTT needed for long simulations. Current validated range: ~100 steps.

### 11.4 Half-Vehicle Bias

A vehicle mid-transition ($w^{(1)} = w^{(2)} = 0.5$) contributes half its presence to each lane. Vehicles behind see a "half-vehicle" with inflated headways. This is a modeling bias — the committed perception fix mitigates for the ego vehicle but not for observers.

---

## 12. Implementation

- **JAX** — automatic differentiation + GPU acceleration
- **Python 3.12** — simulation loop
- **Hardware:** NVIDIA A100 (Azure), RTX 4090 (local)

### Performance

- CPU (N=8, T=1200 steps): forward 9ms, backward 61ms, one optimization step 62ms
- GPU not needed for N=8 paper. CPU is sufficient.

---

## 13. Comparison with Prior Approaches

| Feature | DiffIDM (Son+) | Sigmoid MOBIL (Phase 2) | **This work (Phase 3)** |
|---|---|---|---|
| Lane changes | None | 2-lane, discrete + sigmoid | **N-lane, continuous lateral** |
| Differentiability | Adjoint through IDM | Sigmoid relaxation of MOBIL | **Smooth ODE, no relaxation needed** |
| LC duration | N/A | Instantaneous | **Physical (3–5s)** |
| Multi-lane interaction | N/A | Binary (current/target) | **Continuous (position-based)** |
| Politeness | N/A | Discrete (new/old follower) | **Continuous (marginal impact)** |
| Gradient quality | Exact (no LC) | Approximate (sigmoid) | **Exact (smooth ODE, 10⁻⁹)** |
| Special machinery | None | Saltation + sigmoid + event detection | **None — standard autodiff** |
