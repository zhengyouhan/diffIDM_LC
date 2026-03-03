# ST-Gumbel and Saltation Matrices

Two complementary techniques for differentiating through lane-change events. Each handles a different aspect of the discontinuity.

---

## 1. Straight-Through Gumbel Estimator (ST-Gumbel)

### What It Handles
The **discrete decision**: should vehicle $i$ change lanes or not?

### The Problem
The MOBIL lane-change rule produces a hard binary decision:
$$a_i = \mathbb{1}[h_i(\mathbf{z};\theta) > 0]$$

The indicator function $\mathbb{1}[\cdot]$ has zero gradient almost everywhere — backpropagation through it gives $\partial a_i / \partial \theta = 0$, losing all information about how parameters affect the decision.

### The Solution
Decouple the forward and backward passes:

**Forward (hard):** Use the exact discrete decision. The simulation sees the real, physical lane change.

$$\text{logits} = (0, \; h_i/\tau), \qquad a_i = \arg\max_j(\text{logit}_j + g_j), \quad g_j \sim \text{Gumbel}(0,1)$$

**Backward (soft):** Replace argmax with softmax for gradient computation:

$$y_i^{\text{soft}} = \text{softmax}\left(\frac{\text{logits} + \mathbf{g}}{\tau}\right)$$

The gradient through the decision becomes:

$$\frac{\partial a_i}{\partial h_i} \approx \frac{1}{\tau}\, y_{i,1}^{\text{soft}}\,(1 - y_{i,1}^{\text{soft}})$$

This is the classic straight-through trick (Bengio et al. 2013, Jang et al. 2017).

### Chain to Parameters
Since $h_i$ depends on IDM accelerations which depend on $\theta$:

$$\frac{\partial a_i}{\partial \theta} = \underbrace{\frac{\partial a_i}{\partial h_i}}_{\text{ST gradient}} \cdot \underbrace{\frac{\partial h_i}{\partial \theta}}_{\text{MOBIL → IDM}}$$

### Temperature $\tau$
- $\tau \to 0$: gradient concentrates near $h_i = 0$ (sharp decision boundary, high gradient magnitude)
- $\tau \to \infty$: gradient spreads uniformly (noisy, uninformative)
- We use $\tau = 0.1$ (near-deterministic with slight exploration)

### What ST-Gumbel Captures
How a small change in $\theta$ shifts the **probability** of a lane change occurring.

### What It Does NOT Capture
What happens to the trajectory **after** the decision — the topology change and its dynamical consequences. That requires the saltation matrix.

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

### Structure
$\Xi$ is a **rank-1 perturbation of identity**: $\Xi = I + \mathbf{u}\,\mathbf{w}^\top$

where:
- $\mathbf{u} = \Delta F / (D_\mathbf{z} h \cdot F^-)$ — sparse, at most 3 nonzero entries (lane-changer + affected followers)
- $\mathbf{w} = D_\mathbf{z} h$ — sparse, ~7 nonzero entries (vehicles involved in MOBIL evaluation)

### Adjoint Update
In the backward pass, when we reach a lane-change event:

$$\lambda \gets \Xi^\top \lambda = \lambda + \mathbf{w}\,(\mathbf{u}^\top \lambda)$$

This is $O(N)$, not $O(N^2)$. We never form $\Xi$ as a full matrix.

### Sparsity in Traffic
For a lane change by vehicle $c$:

**$\Delta F$ is nonzero only for:**
- Vehicle $c$ (gets a new leader → different gap → different acceleration)
- Old follower of $c$ (loses $c$ as leader, inherits $c$'s old leader)
- New follower of $c$ (gets $c$ as new leader)

All position components are zero ($\dot{x} = v$ doesn't depend on topology).

### The Grazing Problem
When $D_\mathbf{z} h \cdot F^- \approx 0$ (the trajectory barely crosses the switching surface), $\|\Xi\|$ explodes. This is a **near-tangential lane change** — the vehicle is indifferent and a tiny perturbation determines the outcome.

**Our fix:** Clamp $|D_\mathbf{z} h \cdot F^-| \geq \epsilon_{\text{salt}}$ with $\epsilon_{\text{salt}} = 0.01$. This bounds the saltation correction at the cost of slight gradient bias for grazing events.

---

## 3. How They Compose: The Full Backward Pass

The two techniques are not alternatives — they handle **orthogonal aspects** of the lane-change discontinuity and compose in the backward sweep:

```
λ = ∂J/∂z^{K}          # terminal adjoint
grad_θ = 0

for k = K-1, ..., 0:
    # (A) Adjoint through IDM Euler step
    grad_θ += λᵀ · ∂Φ/∂θ|_k
    λ = (∂Φ/∂z|_k)ᵀ · λ + ∂J_k/∂z^k

    # At lane-change events:
    if k is event j:
        # (B) ST-Gumbel: decision gradient
        γ_j = st_grad_j · (λᵀ · ΔF_j · Δt)
        grad_θ += γ_j · ∂h_j/∂θ

        # (C) Saltation: topology correction
        λ = Ξ_jᵀ · λ
```

### What Each Component Contributes

| Component | Question it answers | Symbol |
|-----------|-------------------|--------|
| **(A) Adjoint** | How does $\theta$ affect the trajectory through smooth IDM dynamics? | $A_k, B_k$ |
| **(B) ST-Gumbel** | How does $\theta$ affect *whether* a lane change occurs? | $\gamma_j \cdot \partial h_j / \partial\theta$ |
| **(C) Saltation** | How does the topology switch affect trajectory sensitivity? | $\Xi_j$ |

### The Cross-Domain Insight
This decomposition is structurally identical to differentiating through **contact events in rigid-body simulation** (Dojo, Drake):

- Saltation matrix = contact Jacobian correction
- ST-Gumbel = smooth contact/separation decision
- Grazing guard = regularized contact models

The traffic and robotics communities face the same mathematical challenge. Our contribution is bringing the hybrid systems machinery to microscopic traffic simulation.

---

## References

1. Jang, Gu, Poole. "Categorical Reparameterization with Gumbel-Softmax." ICLR 2017.
2. Bengio, Léonard, Courville. "Estimating or Propagating Gradients Through Stochastic Neurons." 2013.
3. Burden, Revzen, Sastry. "Event-Selected Vector Field Discontinuities Yield Piecewise-Differentiable Flows." SIAM J. Applied Dynamical Systems, 2016.
4. Vilar, Saiz. "Exact Differentiable Simulation of Stochastic Systems." 2026.
5. Le Cleac'h et al. "Dojo: A Differentiable Physics Engine for Robotics." 2023.
