# References

Key papers for the Differentiable Lane-Change (diffLC) project.

## Core — Car-Following & Lane-Change Models

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Intelligent Driver Model | Treiber, Hennecke, Helbing | 2000 | Car-following model | [DOI](https://doi.org/10.1103/PhysRevE.62.1805) |
| MOBIL | Kesting, Treiber, Helbing | 2007 | Lane-change model | [DOI](https://doi.org/10.3141/2000-12) |
| DiffIDM | Son, Kim, Yun | 2025 | Differentiable IDM for trajectory prediction | [arXiv](https://arxiv.org/abs/2210.15585) |
| IDM 25-Year Survey | Zhou et al. | 2025 | Comprehensive IDM review | — |

## Relaxation of Discrete Decisions

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Categorical Reparameterization with Gumbel-Softmax | Jang, Gu, Poole | 2017 | Gumbel-Softmax estimator | [arXiv](https://arxiv.org/abs/1611.01144) |
| Estimating or Propagating Gradients Through Stochastic Neurons | Bengio, Léonard, Courville | 2013 | Straight-through estimator theory | [arXiv](https://arxiv.org/abs/1308.3432) |
| Exact Differentiable Simulation of Stochastic Systems | Vilar, Saiz | 2026 | ST-Gumbel for CTMCs — project inspiration | [arXiv](https://arxiv.org/abs/2602.19775) |

## Hybrid Systems & Saltation Matrices

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Event-Selected Vector Field Discontinuities Yield Piecewise-Differentiable Flows | Burden, Revzen, Sastry | 2016 | Saltation matrices for hybrid systems | [DOI](https://doi.org/10.1137/15M1016588) |
| Multiple Lyapunov Functions for Switched and Hybrid Systems | Branicky | 1998 | Hybrid dynamical systems foundations | [DOI](https://doi.org/10.1109/9.664150) |
| Adjoint Sensitivity Analysis for Conservation Laws | Ulbrich | 2003 | Adjoint at shocks — macro analog of saltation | [DOI](https://doi.org/10.1007/s10107-003-0421-7) |

## Differentiable Simulation (Cross-Domain)

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Dojo: A Differentiable Physics Engine for Robotics | Le Cleac'h, Howell, Schwager, Manchester | 2023 | Differentiable contact dynamics | [arXiv](https://arxiv.org/abs/2203.00806) |
| Do Differentiable Simulators Give Better Policy Gradients? | Zhong, Kozuno, Peng, van de Panne | 2023 | Gradient correctness analysis | [arXiv](https://arxiv.org/abs/2309.14187) |
| Differentiable Agent-Based Simulation | Andelfinger | 2021 | Differentiable ABM for traffic | [arXiv](https://arxiv.org/abs/2101.02104) |
| A Review of Differentiable Simulators | Newbury et al. | 2024 | Survey of differentiable simulation | — |
| Differentiable Hybrid Traffic Simulation | Son et al. | 2022 | Hybrid micro-macro differentiable traffic sim | — |

## Lane-Change Game Theory

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| A Stackelberg Game Theoretic Model of Lane-Changing | Yoo, Langari | 2020 | Stackelberg games for lane-change | [DOI](https://doi.org/10.1109/TITS.2020.2975008) |
| MPCC Formulation for Lane-Change Interaction | Burger, Zanon, Diehl | 2022 | Bilevel → MPCC, complementarity in LC | [ITSC 2022](https://doi.org/10.1109/ITSC55140.2022.9921807) |
| Unified Risk Field for Lane-Change | Tan et al. | 2024 | Unified potential field for CF+LC | — |
| Potential Field Lane-Change Model | Li et al. | 2022 | Physics-inspired LC model (155 citations) | — |
| Stackelberg Inverse MPC | Zhang et al. | 2024 | Stackelberg + inverse optimal control | — |

## Traffic Estimation & Inverse Problems

| Paper | Authors | Year | Topic | Link |
|-------|---------|------|-------|------|
| Localized Inverse Design in Conservation Laws | Colombo, Perrollaz | 2024 | Inverse problems for traffic PDEs | — |
| Imagining The Road Ahead | Ścibior et al. | 2021 | Differentiable multi-agent trajectory prediction | [arXiv](https://arxiv.org/abs/2005.02550) |
