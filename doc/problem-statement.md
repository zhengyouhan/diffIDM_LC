# Problem Statement

## The Challenge

We want to **differentiate through a microscopic traffic simulator** that includes lane changes. This is needed for gradient-based parameter estimation, trajectory reconstruction, and optimal control in multi-lane traffic.

The simulator consists of two coupled layers:

1. **Car-following (IDM):** Each vehicle $i$ follows the Intelligent Driver Model:

$$\dot{v}_i = a\left[1 - \left(\frac{v_i}{v_0}\right)^\delta - \left(\frac{s^*(v_i, \Delta v_i)}{s_i}\right)^2\right]$$

where $s_i$ is the gap to the leader and $\Delta v_i$ is the approach speed. This is smooth and differentiable.

2. **Lane-changing (MOBIL):** Vehicle $i$ changes lanes when the MOBIL incentive exceeds a threshold:

$$h_i(\mathbf{z}; \theta) = \underbrace{(\tilde{a}_i - a_i)}_{\text{own gain}} + p\underbrace{(\Delta a_{\text{neighbors}})}_{\text{courtesy}} - \Delta a_{\text{th}} > 0$$

This is a **discrete decision** that changes the leader-follower graph — making the system non-differentiable.

## Why Standard Autodiff Fails (The Discrete Formulation)

In the traditional discrete formulation, a lane change is an **instantaneous event** that rewires the interaction topology: vehicle $i$ gets a new leader, its old follower gets a new leader, its new follower gets $i$ as leader. The dynamics operator changes from $F_G$ to $F_{G'}$.

Standard automatic differentiation produces $\partial F_{G'}/\partial\theta$ but misses:

1. **The dynamics jump** $F_{G'} - F_G$ at the switching instant
2. **The timing sensitivity** — how $\theta$ affects *when* the switch occurs
3. **The decision sensitivity** — how $\theta$ affects *whether* the switch occurs

This is the **graph-switching hybrid dynamical system** problem:

$$\dot{\mathbf{z}} = F_{G(\boldsymbol\ell)}(\mathbf{z}; \theta), \qquad G(\boldsymbol\ell) = \text{leader-follower graph from lane assignments } \boldsymbol\ell$$

Structurally identical to **contact events in rigid-body simulation** (Dojo, Drake):

| Traffic | Rigid-body |
|---------|-----------|
| Lane change (leader switch) | Contact on/off |
| Minimum spacing $s \geq s_0$ | Non-penetration $\phi \geq 0$ |
| MOBIL incentive $h > 0$ → switch | Contact force $\lambda > 0$ → push |
| Saltation matrix $\Xi$ | Saltation matrix $\Xi$ |

## Our Solution: Eliminate the Discontinuity

Rather than engineering gradients through discrete events (saltation matrices, sigmoid relaxation — which we explored in Phase 1–2), we **remove the discontinuity entirely** by replacing the discrete lane-change formulation with **continuous lateral dynamics**.

### The Key Insight

Lane changes in reality are not instantaneous. A vehicle physically moves sideways over 3–5 seconds, during which it interacts with vehicles in multiple lanes based on its physical lateral position. By modeling this continuous lateral movement directly, the entire simulation becomes a smooth ODE — no discrete events, no topology switches, no special gradient machinery.

### Vehicle State

Each vehicle $i$ has continuous state:

$$\mathbf{z}_i = (x_i, v_i, y_i) \in \mathbb{R}^3$$

- $x_i$: longitudinal position
- $v_i$: longitudinal velocity
- $y_i$: **lateral position** (continuous, not a lane index)

Lane membership is determined by physical proximity:

$$w_i^{(k)} = \frac{\exp\left(-\gamma(y_i - y_k)^2\right)}{\sum_{k'} \exp\left(-\gamma(y_i - y_{k'})^2\right)}$$

where $y_k$ are lane centers and $\gamma$ controls sharpness.

### The System Is One Smooth ODE

$$\dot{x}_i = v_i$$
$$\dot{v}_i = a_i(x, v, y;\, \theta) \quad \text{(IDM with position-based blended headway)}$$
$$\dot{y}_i = u_{\max} \cdot \tanh\left(\frac{\kappa(y_i^{\text{target}} - y_i)}{u_{\max}}\right) \quad \text{(lateral dynamics toward MOBIL target)}$$

**No discrete events.** No event detection. No graph rewiring. No saltation matrices. Just a smooth ODE that can be differentiated with standard reverse-mode AD.

Lane changes happen when $\dot{y}_i \neq 0$, which occurs whenever the MOBIL softmax prefers a different lane. The vehicle physically moves, its lane weights shift continuously, and the interaction graph evolves smoothly.

## The Inverse Problems

We consider two inverse problems:

### 1. IDM Parameter Estimation (macro → micro)
Given macroscopic observations (density, flow from loop detectors), recover IDM parameters $\theta = (v_0, T, a, b)$ by minimizing:

$$J(\theta) = \| \rho^{\text{sim}}(\theta) - \rho^{\text{obs}} \|^2 + \| q^{\text{sim}}(\theta) - q^{\text{obs}} \|^2$$

where macroscopic fields are obtained from microscopic trajectories via Gaussian kernel smoothing.

### 2. Trajectory Reconstruction (micro → micro)
Given observed vehicle trajectories, recover IDM and MOBIL parameters:

$$J(\theta) = \sum_{i,k}\left[\alpha_x(x_i^k - \hat{x}_i^k)^2 + \alpha_v(v_i^k - \hat{v}_i^k)^2 + \alpha_y(y_i^k - \hat{y}_i^k)^2\right]$$

The lateral error term $\alpha_y(y_i^k - \hat{y}_i^k)^2$ is unique to the continuous formulation — it provides direct gradient signal for lane-change-related parameters.

Both require $dJ/d\theta$, computed via standard reverse-mode autodiff through the smooth ODE.

## Why Not Just Fix the Gradients? (Phase 1–2 Lessons)

We initially pursued two approaches to differentiate through the discrete formulation:

| Phase | Approach | Outcome |
|-------|----------|---------|
| Phase 1 | ST-Gumbel-Softmax + saltation matrices | Gumbel gradients saturate on decisive MOBIL decisions ($h \gg 0$) |
| Phase 2 | Sigmoid relaxation ($\sigma(h/\tau)$) + saltation | Works — 11.1% error at $\tau=0.1$. But still approximate. |

Phase 2 proved that **decision sensitivity is essential** (zero gradient without sigmoid), and that **saltation is secondary** in sparse-LC regimes. But both phases require:
- Custom backward pass logic
- Careful temperature scheduling
- Saltation matrix computation at each event
- Event detection and graph management

The continuous lateral dynamics formulation achieves **exact gradients** (matching FD to 10⁻⁹) with **none of this machinery** — just standard JAX autodiff through a smooth ODE.

→ See [methodology.md](methodology.md) for the full current approach.
→ See [gradient-techniques.md](gradient-techniques.md) for detailed exposition of all three phases.
