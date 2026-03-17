# Gradient Techniques for Lane-Change Discontinuities

Evolution of our approach to differentiating through lane-change events, from discrete-event methods (Phases 1–2) to the current continuous formulation (Phase 3).

---

## Phase 3 (Current): Continuous Lateral Dynamics — No Special Machinery

### Core Idea

By replacing discrete lane-change events with continuous lateral movement ($y_i(t) \in \mathbb{R}$), the entire simulation becomes a smooth ODE. Lane membership, leader selection, and inter-vehicle interactions are determined by physical position through smooth functions (Gaussian weights, log-sum-exp headways, softmax decisions, tanh saturation).

### What Makes It Differentiable

Every component is $C^\infty$:

| Component | Function | Smoothness |
|---|---|---|
| Lane weights | Softmax of Gaussian | $C^\infty$ |
| Headway | Log-sum-exp soft-min | $C^\infty$ |
| IDM | Standard IDM with softplus $s^*$ | $C^\infty$ |
| Decision | Softmax over MOBIL incentives | $C^\infty$ |
| Lateral velocity | tanh saturation | $C^\infty$ |

### Gradient Method

Standard reverse-mode AD (JAX `jax.grad`) through unrolled Euler integration. No custom adjoint, no event handling, no saltation. Just the chain rule.

### Validated Quality

All IDM parameters match finite differences to **8–10 significant digits** with 5 active lane changes in the scenario.

### Why This Works

The key insight: in the discrete formulation, the non-differentiability comes from instantaneous topology changes. In reality, lane changes are 3–5 second physical movements. By modeling the physics correctly (continuous lateral position), the differentiability problem disappears — it was an artifact of the discrete abstraction, not of the physics.

---

## Phase 2 (Historical): Sigmoid Relaxation (Logit-MOBIL)

### What It Handles

The **discrete decision**: should vehicle $i$ change lanes or not?

### The Problem

The MOBIL rule produces a hard binary decision:
$$a_i = \mathbb{1}[h_i(\mathbf{z};\theta) > 0]$$

The indicator function has zero gradient almost everywhere.

### The Solution

Replace with sigmoid:
$$P(\text{LC}_i) = \sigma(h_i / \tau) = \frac{1}{1 + e^{-h_i/\tau}}$$

### Temperature $\tau$

- $\tau \to 0$: approaches hard step (physical, but gradient concentrates near $h = 0$)
- $\tau \to \infty$: flattens (smooth everywhere, but unphysical)
- **$\tau = 0.1$ optimal**: nearly binary + smooth gradient

### Results

- Hard MOBIL + saltation: **30.6%** mean parameter error
- Sigmoid relaxation ($\tau = 0.1$): **11.1%** mean error
- Without sigmoid (configs C, D): **exactly zero gradient** for MOBIL parameters

### Why Not Gumbel-Softmax?

Gumbel-Softmax failed because MOBIL decisions are typically **decisive** — the incentive $h$ is far from zero, causing gradient saturation. Works for 3+ lanes where decisions are genuinely uncertain (at least one pair of probabilities near 0.5).

### Status

**Validated but superseded.** The sigmoid approach proved that decision sensitivity is essential and saltation is secondary. Phase 3 achieves exact gradients without any relaxation.

---

## Phase 1 (Historical): ST-Gumbel-Softmax + Saltation Matrices

### The Approach

1. **Gumbel-Softmax** for the discrete lane-change decision (Jang et al. 2017)
2. **Saltation matrices** for the dynamics jump at topology switches (Burden et al. 2016)

### Saltation Matrix

From hybrid systems theory, the first-order sensitivity correction at a switching surface $h(\mathbf{z}) = 0$:

$$\Xi = I + \frac{\Delta F \cdot (D_\mathbf{z} h)^\top}{D_\mathbf{z} h \cdot F^-}$$

where:
- $\Delta F = F_{G'} - F_G$: jump in dynamics
- $D_\mathbf{z} h$: gradient of MOBIL incentive w.r.t. state
- $F^- = F_G(\mathbf{z}^*)$: pre-switch dynamics
- $D_\mathbf{z} h \cdot F^-$: crossing speed

Adjoint update: $\lambda \gets \Xi^\top \lambda$ at each LC event. $O(N)$ — never form $\Xi$ as a full matrix.

### Why It Failed

Gumbel-Softmax gradients saturate on decisive MOBIL decisions. When $h \gg 0$ or $h \ll 0$, the softmax output is near 0 or 1, and the straight-through gradient vanishes. MOBIL decisions in typical traffic are almost always decisive — the incentive is clearly positive or negative.

### What We Learned

- Saltation reduces IDM estimation error in dense traffic (33.7% → 30.7%)
- Saltation is negligible with sparse LC events (configs A ≈ B in ablation)
- The **grazing problem** ($D_\mathbf{z} h \cdot F^- \approx 0$) causes $\|\Xi\|$ explosion. Clamping fixes it numerically but the issue is fundamental.
- **Why saltation is structurally limited for traffic (Mar 11 finding):** Lane changes are NOT a genuine hybrid system — same IDM dynamics before and after, just different inputs (headway). Saltation needs genuinely different dynamics ($f^- \neq f^+$). MOBIL safety keeps $\Delta F$ small (~0.4 m/s²). The correct application of saltation in traffic would be free-flow ↔ congested transitions, not individual lane changes.

### Status

**Abandoned.** Gumbel gradients fundamentally incompatible with decisive binary decisions. Saltation addresses the wrong problem for individual lane changes.

---

## How The Phases Compare

| | Phase 1 | Phase 2 | **Phase 3** |
|---|---|---|---|
| **Decision gradient** | Gumbel-Softmax (saturates) | Sigmoid (works) | Softmax over continuous target (exact) |
| **Topology gradient** | Saltation matrix | Saltation matrix | Not needed (smooth ODE) |
| **IDM gradient** | JAX adjoint | JAX adjoint | JAX adjoint |
| **Best result** | Failed | 11.1% error | 10⁻⁹ FD match |
| **Machinery required** | Event detection + Gumbel + saltation | Event detection + sigmoid + saltation | Standard autodiff |
| **Lane count** | 2 | 2 | Arbitrary $K$ |
| **LC model** | Instantaneous switch | Instantaneous switch | Physical 3–5s movement |

---

## Key Lessons

1. **The gradient technique must match the decision structure.** Binary MOBIL → sigmoid. Multi-choice → Gumbel-Softmax. But the deepest fix is eliminating the discrete decision entirely.

2. **Decision sensitivity dominates topology sensitivity.** In Phase 2 ablation, removing sigmoid gradient → 80% error; removing saltation → negligible change. This means the *whether* of a lane change matters more than the *dynamics jump*.

3. **The discontinuity was an artifact, not physics.** Real lane changes are continuous. The discrete formulation was a modeling convenience that created an artificial differentiability barrier.

4. **Saltation has a limited role in traffic.** Lane changes don't produce genuinely different dynamics (same IDM, different inputs). The correct domain for saltation in traffic is regime transitions (free-flow ↔ congested), not individual lane changes.

---

## References

1. Jang, Gu, Poole. "Categorical Reparameterization with Gumbel-Softmax." ICLR 2017.
2. Bengio, Léonard, Courville. "Estimating or Propagating Gradients Through Stochastic Neurons." 2013.
3. Burden, Revzen, Sastry. "Event-Selected Vector Field Discontinuities Yield Piecewise-Differentiable Flows." SIAM J. Applied Dynamical Systems, 2016.
4. Le Cleac'h et al. "Dojo: A Differentiable Physics Engine for Robotics." 2023.
5. Li et al. "Incremental Potential Contact: Intersection- and Inversion-Free Large-Deformation Dynamics." ACM TOG, 2020.
