# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository re-implements papers in the **multi-objective optimization (MOO)** topic, with a focus on gradient-based methods applied to machine learning. Each subdirectory corresponds to one paper/method and is largely self-contained.

The methods differ across multiple dimensions:

- **Problem formulation** — some target a single Pareto-optimal point (given a preference vector), others find a diverse set of solutions covering the Pareto front
- **Number of solutions** — single-solution vs. population/set-based output
- **Solver type** — LP, min-norm (Frank-Wolfe), KKT/Hessian, hypernetwork, etc.
- **Application domain** — multi-task learning, multi-target regression, multi-label classification, toy synthetic problems

| Directory | Method | Paper | Solutions |
|-----------|--------|-------|-----------|
| `CPMTL/` | Continuous Pareto MTL | Ma et al., ICML 2020 | Set (continuous exploration) |
| `EPOSearch/` | Exact Pareto Optimal Search | Mahapatra & Rajan, ICML 2020 | Single (preference-specific) |
| `PMTL/` | Pareto Multi-Task Learning | Lin et al., NeurIPS 2019 | Set (preference-guided) |
| `MOO-MTL/` | MOO as MTL (MGDA_UB) | Sener & Koltun, NeurIPS 2018 | Single (Pareto stationary) |
| `PHN/` | Preference-based Hypernetwork | — | Continuous (hypernetwork) |

## Architecture

### Common algorithmic pattern

All methods share the same conceptual loop:
1. Forward pass → compute per-task losses
2. Compute per-task gradients
3. Solve a convex subproblem over the gradients (method-specific solver)
4. Apply the resulting descent direction to update shared parameters

The core algorithmic difference between methods lives entirely in step 3.

### Gradient solvers

| Method | Solver | Algorithm |
|--------|--------|-----------|
| CPMTL | `solvers/kkt_solver.py`, `hvp_solver.py` | KKT + Hessian-vector products |
| EPOSearch | `toy_experiments/solvers/epo_lp.py` | LP-based EPO |
| PMTL | `solvers/min_norm_solvers.py` | Min-norm gradient (Frank-Wolfe) |
| MOO-MTL | `solvers/min_norm_solver.py` | MGDA_UB |
| PHN | `solvers.py` | Preference-conditioned hypernetwork |

