# NumPy Neural Network

A small multi-layer classifier built entirely from matrix math. No PyTorch, no TensorFlow, no autograd. Just NumPy, the chain rule, and gradient descent.

## What this is

The `NeuralNetwork` class wires up a fully connected stack of layers from an architecture list like `[784, 128, 64, 10]`. Hidden layers use ReLU. The output layer uses softmax with cross-entropy loss. Weights are initialized with He (Kaiming) scaling so early activations stay in a sensible range.

Forward pass, backward pass, and parameter updates are all written out explicitly:

- Forward: `Z = W @ A + b`, then ReLU or softmax
- Loss: cross-entropy against one-hot labels
- Backward: layer-by-layer gradients via the chain rule
- Update: plain batch gradient descent

## Setup

```bash
pip install -r requirements.txt
```

You only need NumPy.

## Run the demos

```bash
python train.py
```

The script runs two checks:

1. **Two moons**: a synthetic binary dataset with interleaved crescents. It is not linearly separable, so it is a clean check that the network can learn a curved decision boundary.

2. **MNIST**: handwritten digits downloaded on first run into a local `data/` folder. Training uses a subset by default so it finishes in a reasonable time on a laptop.

During training you will see loss and accuracy printed at regular intervals. Both should improve as epochs go on.

## Using the class yourself

```python
from neural_network import NeuralNetwork

model = NeuralNetwork([input_dim, hidden_dim, num_classes])
history = model.fit(X_train, y_train, epochs=500, learning_rate=0.1)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

`X` can be shaped as `(num_samples, num_features)` or `(num_features, num_samples)` (the class normalizes internally). Labels should be integer class indices.

## How the math fits together

For a batch of `m` samples stored as columns:

| Step | Formula |
|------|---------|
| Linear layer | `Z^[l] = W^[l] A^[l-1] + b^[l]` |
| ReLU | `A^[l] = max(0, Z^[l])` |
| Softmax | `A^[L] = exp(Z^[L] - max) / sum(exp(...))` |
| Loss | `L = -1/m sum(y log A^[L])` |
| Output gradient | `dZ^[L] = A^[L] - Y` |
| Hidden gradient | `dZ^[l] = (W^[l+1]^T dZ^[l+1]) ⊙ ReLU'(Z^[l])` |
| Weight gradient | `dW^[l] = 1/m dZ^[l] (A^[l-1])^T` |
| Bias gradient | `db^[l] = 1/m sum(dZ^[l])` |

Softmax subtracts the per-column maximum before exponentiating to avoid overflow. Cross-entropy clips probabilities slightly so `log(0)` never appears.

The output-layer gradient simplifies to `A - Y` when softmax and cross-entropy are paired. That is the usual trick and it is what the backward pass implements.

## Project layout

```
numpy-neural-network/
├── neural_network.py   # The network class
├── train.py            # Two-moons + MNIST verification
├── requirements.txt
└── README.md
```

## Notes

This is educational code. There is no mini-batch shuffling, momentum, Adam, dropout, or regularization. Those are straightforward to add once the core loops make sense. The goal here is to show that a working classifier really is just repeated matrix multiply, a nonlinearity, and careful calculus on the way back.
