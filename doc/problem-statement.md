# Problem Statement

## The Challenge

We want to **differentiate through a microscopic traffic simulator** that includes lane changes. This is needed for gradient-based parameter estimation, trajectory reconstruction, and optimal control in multi-lane traffic.

The simulator consists of two layers:

1. **Continuous layer (IDM car-following):** Each vehicle $i$ follows the Intelligent Driver Model:

$$\dot{v}_i = a\left[1 - \left(\frac{v_i}{v_0}\right)^\delta - \left(\frac{s^*(v_i, \Delta v_i)}{s_i}\right)^2\right]$$

where $s_i$ is the gap to the leader and $\Delta v_i$ is the approach speed. This is smooth and differentiable.

2. **Discrete layer (MOBIL lane-changing):** Vehicle $i$ changes lanes when the MOBIL incentive exceeds a threshold:

$$h_i(\mathbf{z}; \theta) = \underbrace{(\tilde{a}_i - a_i)}_{\text{own gain}} + p\underbrace{(\Delta a_{\text{neighbors}})}_{\text{courtesy}} - \Delta a_{\text{th}} > 0$$

This is a **discrete decision** that changes the leader-follower graph.

## Why Standard Autodiff Fails

When a lane change occurs, the **interaction topology switches**: vehicle $i$ gets a new leader, its old follower gets a new leader, its new follower gets $i$ as leader. The dynamics operator changes from $F_G$ to $F_{G'}$.

Standard automatic differentiation (e.g., JAX, PyTorch) computes gradients assuming the computation graph is fixed. It produces $\partial F_{G'}/\partial\theta$ when the correct sensitivity must account for:

1. **The dynamics jump** $F_{G'} - F_G$ at the switching instant
2. **The timing sensitivity** — how $\theta$ affects *when* the switch occurs
3. **The decision sensitivity** — how $\theta$ affects *whether* the switch occurs

Ignoring these produces **biased gradients** that can mislead optimization.

## The System as a Hybrid Dynamical System

The traffic simulator is a **graph-switching hybrid dynamical system**:

$$\dot{\mathbf{z}} = F_{G(\boldsymbol\ell)}(\mathbf{z}; \theta), \qquad G(\boldsymbol\ell) = \text{leader-follower graph determined by lane assignments } \boldsymbol\ell$$

Lane changes are **guard-triggered discrete transitions**: when $h_i(\mathbf{z};\theta)$ crosses zero from below, the lane assignment $\ell_i$ flips and the graph $G$ changes.

This is structurally identical to **contact events in rigid-body simulation**:

| Traffic | Rigid-body |
|---------|-----------|
| Lane change (leader switch) | Contact on/off |
| Minimum spacing $s \geq s_0$ | Non-penetration $\phi \geq 0$ |
| MOBIL incentive $h > 0$ → switch | Contact force $\lambda > 0$ → push |
| Complementarity: $\mu \perp g \geq 0$ | Same NCP structure |
| Saltation matrix $\Xi$ | Saltation matrix $\Xi$ |

The robotics community has developed scalable methods for differentiating through contact (Dojo, Drake). **Nobody has applied this to traffic.**

## The Inverse Problem

Given observed trajectories $\mathbf{z}^{\text{obs}}$, recover the IDM parameters $\theta = (v_0, T, a, b)$ by minimizing:

$$J(\theta) = \frac{1}{NK}\sum_{i,k}\left(x_i^{\text{sim}}(k;\theta) - x_i^{\text{obs}}(k)\right)^2$$

This requires $dJ/d\theta$, which must correctly propagate through all lane-change events in the simulation.

## What We Need

A backward pass that composes three gradient components:

1. **Adjoint through IDM dynamics** (smooth arcs between lane changes)
2. **Saltation matrix correction** at each topology switch (continuous dynamics jump)
3. **Straight-Through Gumbel gradient** through the discrete lane-change decision

→ See [methodology.md](methodology.md) for our solution.
