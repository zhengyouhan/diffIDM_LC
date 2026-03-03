# References

Key papers for the Differentiable Lane-Change (diffLC) project.

## Core Methodology

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Intelligent Driver Model | Treiber, Hennecke, Helbing | 2000 | Car-following model | [DOI](https://doi.org/10.1103/PhysRevE.62.1805) |
| MOBIL | Kesting, Treiber, Helbing | 2007 | Lane-change model | [DOI](https://doi.org/10.3141/2000-12) |
| DiffIDM | Son, Kim, Yun | 2025 | Differentiable IDM for trajectory prediction | [arXiv](https://arxiv.org/abs/2210.15585) |
| Categorical Reparameterization with Gumbel-Softmax | Jang, Gu, Poole | 2017 | Straight-through Gumbel estimator | [arXiv](https://arxiv.org/abs/1611.01144) |
| Estimating or Propagating Gradients Through Stochastic Neurons | Bengio, Léonard, Courville | 2013 | Straight-through estimator theory | [arXiv](https://arxiv.org/abs/1308.3432) |
| Exact Differentiable Simulation of Stochastic Systems | Vilar, Saiz | 2026 | ST-Gumbel for CTMCs — our inspiration | [arXiv](https://arxiv.org/abs/2602.19775) |

## Hybrid Systems & Saltation Matrices

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Event-Selected Vector Field Discontinuities Yield Piecewise-Differentiable Flows | Burden, Revzen, Sastry | 2016 | Saltation matrices for hybrid systems | [DOI](https://doi.org/10.1137/15M1016588) |
| Stability and Robustness Analysis of Switched and Hybrid Systems | Branicky | 1998 | Hybrid dynamical systems foundations | [DOI](https://doi.org/10.1109/9.664150) |
| Adjoint Sensitivity Analysis for Conservation Laws (Ulbrich) | Ulbrich | 2003 | Adjoint at shocks — macro analog of saltation | [DOI](https://doi.org/10.1007/s10107-003-0421-7) |

## Differentiable Contact (Cross-domain)

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Dojo: A Differentiable Physics Engine for Robotics | Le Cleac'h, Howell, Schwager, Manchester | 2023 | Differentiable contact via single-level NCP | [arXiv](https://arxiv.org/abs/2203.00806) |
| How to Train Your Differentiable Simulation | Zhong, Kozuno, Peng, van de Panne | 2023 | Gradient correctness in differentiable sim | [arXiv](https://arxiv.org/abs/2309.14187) |
| Differentiable Agent-Based Simulation for Gradient-Guided Simulation-Based Optimization | Andelfinger | 2021 | Differentiable ABM | [arXiv](https://arxiv.org/abs/2101.02104) |

## Lane-Change Game Theory & Complementarity

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| A Stackelberg Game Theoretic Model of Lane-Changing | Yoo, Langari | 2020 | Stackelberg games for lane-change | [DOI](https://doi.org/10.1109/TITS.2020.2975008) |
| MPCC Formulation for Lane-Change Interaction | Burger, Zanon, Diehl | 2022 | Bilevel → MPCC, complementarity in LC | [ITSC 2022](https://doi.org/10.1109/ITSC55140.2022.9921807) |
| Differentiable Equilibrium Computation for Stackelberg Congestion Games | Sakaue, Nakamura | 2021 | Differentiable Stackelberg equilibrium | [NeurIPS 2021](https://arxiv.org/abs/2106.00362) |

## Traffic Estimation (Downstream Application)

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Road Traffic Reconstruction (Colombo & Perrollaz) | Colombo, Perrollaz | 2024 | Traffic state estimation survey | — |
| Imagining The Road Ahead | Ścibior et al. | 2021 | Multi-agent trajectory prediction | [arXiv](https://arxiv.org/abs/2005.02550) |
