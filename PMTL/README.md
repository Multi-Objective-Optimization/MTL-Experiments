# Pareto Multi-Task Learning

Implementation of the NeurIPS 2019 paper [Pareto Multi-Task Learning](https://papers.nips.cc/paper/9374-pareto-multi-task-learning).

ParetoMTL treats multi-task learning as a multi-objective optimization problem and finds a set of Pareto-optimal solutions, each corresponding to a different trade-off between tasks. A set of preference vectors guides the optimizer toward different regions of the Pareto front, producing diverse and well-distributed solutions.

---

## Project Structure

```
ParetoMTL/
├── configs/
│   ├── train.yaml               # Training configuration
│   └── synthetic.yaml           # Synthetic example configuration
├── models/
│   ├── lenet.py                 # LeNet backbone + MTL trainer wrapper
│   └── resnet.py                # ResNet18 backbone + MTL trainer wrapper
├── solvers/
│   ├── min_norm_solvers.py      # PyTorch min-norm solver (MOO-MTL)
│   └── min_norm_solvers_numpy.py# NumPy min-norm solver
├── utils/
│   └── preference.py            # Preference vector generation
├── train.py                     # Main training script
├── run_synthetic_example.py     # Synthetic 2D demo
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Synthetic Example

Demonstrates ParetoMTL on a 2D multi-objective problem and plots the found solutions against the true Pareto front.

```bash
python run_synthetic_example.py configs/synthetic.yaml
```

Edit `configs/synthetic.yaml` to switch method or number of solutions:

```yaml
method: ParetoMTL       # ParetoMTL | MOOMTL | Linear
num: 10
```

### Training on Multi-Task Image Classification

```bash
python train.py configs/train.yaml
```

Edit `configs/train.yaml` to configure the experiment:

```yaml
dataset: mnist          # mnist | fashion | fashion_and_mnist
model: lenet            # lenet | resnet18
niter: 100
npref: 5
data_dir: data
save_dir: saved_model
```

**Datasets** — place the corresponding `.pickle` files under `data/`:

| `dataset` value      | File                          |
|----------------------|-------------------------------|
| `mnist`              | `multi_mnist.pickle`          |
| `fashion`            | `multi_fashion.pickle`        |
| `fashion_and_mnist`  | `multi_fashion_and_mnist.pickle` |

Trained models are saved to `save_dir` as `{dataset}_{model}_{niter}_npref{npref}_pref{idx}.pkl`.

---

## How It Works

ParetoMTL proceeds in two phases for each preference vector:

1. **Initialization** — finds a feasible starting point that satisfies the preference constraint.
2. **Main training** — optimizes toward the Pareto-optimal solution corresponding to the given preference vector.

At each step, the gradient direction is computed by solving a quadratic program (via `MinNormSolver`) over the task gradients, ensuring the update moves toward the target Pareto region.

---

## Citation

```bibtex
@inproceedings{lin2019pareto,
  title={Pareto Multi-Task Learning},
  author={Lin, Xi and Zhen, Hui-Ling and Li, Zhenhua and Zhang, Qingfu and Kwong, Sam},
  booktitle={Thirty-third Conference on Neural Information Processing Systems (NeurIPS)},
  pages={12037--12047},
  year={2019}
}
```
