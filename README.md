# Deep Learning from Scratch 🧠

A modular, NumPy-only deep learning framework designed for educational exercises and building core intuition about forward propagation, backpropagation, gradient calculation, optimizers, and neural network architectures.

---

## 📁 Directory Structure

```text
dl_from_scratch/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
├── exercises/
│   ├── 01_tensor_operations.py
│   ├── 02_linear_regression.py
│   ├── 03_logistic_regression.py
│   ├── 04_mlp_backpropagation.py
│   └── 05_train_mnist.py
├── notebooks/
│   ├── 01_autograd_scratch.ipynb
│   └── 02_neural_net_scratch.ipynb
├── src/
│   ├── __init__.py
│   ├── activations/
│   │   ├── __init__.py
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   └── softmax.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   └── dataset.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── conv2d.py
│   │   ├── dense.py
│   │   ├── dropout.py
│   │   └── flatten.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cross_entropy.py
│   │   └── mse.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── sequential.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── adam.py
│   │   ├── base.py
│   │   └── sgd.py
│   └── utils/
│       ├── __init__.py
│       ├── grad_check.py
│       ├── metrics.py
│       └── plot.py
└── tests/
    ├── __init__.py
    ├── grade.py               # Central autograder runner (100 pts)
    ├── grader_utils.py        # Numerical gradient checker & runner utilities
    ├── test_activations.py    # Part 1 tests (15 pts)
    ├── test_layers.py         # Part 2 tests (30 pts)
    ├── test_losses.py         # Part 3 tests (25 pts)
    └── test_optimizers.py     # Part 4 tests (30 pts)
```

---

## 🧪 Grading & Verification

You can check your progress and grade your implementations at any time!

### Run All Parts & View Scorecard (100 Pts Total)
```bash
python tests/grade.py
```

### Run Individual Parts
```bash
# Part 1: Activations (15 pts)
python tests/test_activations.py

# Part 2: Layers (30 pts)
python tests/test_layers.py

# Part 3: Losses (25 pts)
python tests/test_losses.py

# Part 4: Optimizers (30 pts)
python tests/test_optimizers.py
```

Or using `pytest`:
```bash
pytest tests/
```
