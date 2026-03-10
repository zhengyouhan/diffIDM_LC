# Gradient Techniques for Lane-Change Discontinuities

Three techniques for differentiating through lane-change events. Each handles a different aspect of the discontinuity.

---

## 1. Sigmoid Relaxation (Logit-MOBIL)

### What It Handles
The **discrete decision**: should vehicle $i$ change lanes or not?

### The Problem
The MOBIL lane-change rule produces a hard binary decision:
$$a_i = \mathbb{1}[h_i(\mathbf{z};\theta) > 0]$$

The indicator function $\mathbb{1}[\cdot]$ has zero gradient almost everywhere — backpropagation through it gives $\partial a_i / \partial \theta = 0$, losing all information about how parameters affect the decision.

### The Solution: Sigmoid Relaxation
Replace the hard threshold with a sigmoid:

$$P(\text{LC}_i) = \sigma(h_i / \tau) = \frac{1}{1 + e^{-h_i/\tau}}$$

The gradient through the decision becomes:

$$\frac{\partial P}{\partial h_i} = \frac{1}{\tau}\, \sigma(h_i/\tau)\,(1 - \sigma(h_i/\tau))$$

### Temperature $\tau$
- $\tau \to 0$: sigmoid approaches hard step function (physical, but gradient concentrates near $h_i = 0$)
- $\tau \to \infty$: sigmoid flattens (smooth everywhere, but unphysical lane changes)
- **$\tau = 0.1$ is optimal**: nearly binary decisions with smooth gradient

### Why Not Gumbel-Softmax?
We initially tried the straight-through Gumbel estimator (Jang et al. 2017). It failed because MOBIL decisions are typically **decisive** — the incentive $h$ is far from zero, causing gradient saturation. Gumbel-Softmax works best when decisions are genuinely uncertain (probabilities near 0.5), which only occurs in multi-lane scenarios with 3+ options.

**Empirical evidence:** In ablation, configs without sigmoid relaxation (C, D) produce **exactly zero gradient** for MOBIL parameters. The sigmoid is the essential ingredient.

### When Gumbel Does Work
For 3+ lane scenarios, the decision becomes a multi-way choice (left/stay/right) via Gumbel-Softmax over logits $(h_{\text{left}}, 0, h_{\text{right}})$. With 3 options, at least one pair of probabilities is genuinely uncertain, and Gumbel gradients survive. This is tested on the `feature/multi-lane-gumbel` branch.

---

## 2. Saltation Matrices

### What They Handle
The **continuous dynamics jump** when the leader-follower graph switches.

### The Problem
At a lane-change event, the dynamics operator changes: $F_G \to F_{G'}$. A perturbation $\delta\mathbf{z}$ in the pre-event state maps to a different perturbation in the post-event state because:

1. The dynamics $F$ changes (new leader → different gap, different acceleration)
2. The event **timing** shifts (a perturbed trajectory crosses the switching surface at a different time)

The standard adjoint ignores both effects.

### The Solution: Saltation Matrix
From hybrid systems theory (Burden et al. 2016), the first-order sensitivity correction at a switching surface $h(\mathbf{z}) = 0$ is:

$$\Xi = I + \frac{\Delta F \cdot (D_\mathbf{z} h)^\top}{D_\mathbf{z} h \cdot F^-}$$

where:

| Symbol | Meaning |
|--------|---------|
| $\Delta F = F_{G'} - F_G$ | Jump in dynamics at the switching instant |
| $D_\mathbf{z} h$ | Gradient of MOBIL incentive w.r.t. state |
| $F^- = F_G(\mathbf{z}^*)$ | Pre-switch dynamics |
| $D_\mathbf{z} h \cdot F^-$ | "Crossing speed" — how fast the incentive crosses zero |

### Adjoint Update
In the backward pass, when we reach a lane-change event:

$$\lambda \gets \Xi^\top \lambda = \lambda + \mathbf{w}\,(\mathbf{u}^\top \lambda)$$

This is $O(N)$ — we never form $\Xi$ as a full matrix.

### Empirical Findings
- Saltation **matters** for IDM parameter estimation in dense traffic (reduces error from 33.7% → 30.7%)
- Saltation has **negligible effect** on MOBIL parameter reconstruction with sparse LC events (configs A ≈ B)
- Hypothesis: saltation becomes more important as the number of LC events increases. Congested-flow ablation pending.

### The Grazing Problem
When $D_\mathbf{z} h \cdot F^- \approx 0$, $\|\Xi\|$ explodes. **Fix:** Clamp $|D_\mathbf{z} h \cdot F^-| \geq 0.01$.

---

## 3. Adjoint Through IDM Dynamics

Standard reverse-mode AD through the smooth car-following arcs between lane-change events:

$$(\lambda_\mathbf{z}, \lambda_\theta) = \text{vjp}(\Phi_G, \mathbf{z}^k, \theta)^\top(\lambda)$$

This is handled natively by JAX's autodiff. No special treatment needed — IDM is smooth and differentiable.

---

## 4. How They Compose: The Full Backward Pass

```
λ = ∂J/∂z^{K}          # terminal adjoint
grad_θ = 0

for k = K-1, ..., 0:
    # (A) Adjoint through IDM Euler step
    grad_θ += λᵀ · ∂Φ/∂θ|_k
    λ = (∂Φ/∂z|_k)ᵀ · λ + ∂J_k/∂z^k

    # At lane-change events:
    if k is event j:
        # (B) Sigmoid gradient: decision sensitivity
        γ_j = σ'(h_j/τ) · (λᵀ · ΔF_j · Δt)
        grad_θ += γ_j · ∂h_j/∂θ

        # (C) Saltation: topology correction
        λ = Ξ_jᵀ · λ
```

### What Each Component Contributes

| Component | Question it answers |
|-----------|-------------------|
| **(A) Adjoint** | How does $\theta$ affect the trajectory through smooth IDM dynamics? |
| **(B) Sigmoid** | How does $\theta$ affect *whether* a lane change occurs? |
| **(C) Saltation** | How does the topology switch affect trajectory sensitivity? |

---

## 5. Evolution of Approach

| Phase | Approach | Result |
|-------|----------|--------|
| Phase 1 | ST-Gumbel + saltation | Gumbel gradients saturate on decisive MOBIL decisions |
| Phase 2 | **Sigmoid relaxation + saltation** | **11.1% error** (τ=0.1), gradient flows through LC decisions |
| Future | Multi-lane Gumbel-Softmax | 3-way softmax works when decisions are genuinely uncertain |

The key lesson: the gradient technique must match the decision structure. Binary MOBIL → sigmoid. Multi-choice (3+ lanes) → Gumbel-Softmax.

---

## References

1. Jang, Gu, Poole. "Categorical Reparameterization with Gumbel-Softmax." ICLR 2017.
2. Bengio, Léonard, Courville. "Estimating or Propagating Gradients Through Stochastic Neurons." 2013.
3. Burden, Revzen, Sastry. "Event-Selected Vector Field Discontinuities Yield Piecewise-Differentiable Flows." SIAM J. Applied Dynamical Systems, 2016.
4. Vilar, Saiz. "Exact Differentiable Simulation of Stochastic Systems." 2026.
5. Le Cleac'h et al. "Dojo: A Differentiable Physics Engine for Robotics." 2023.
