# Lightweight Neural Network for Sensor Anomaly Detection

## Central Engineering Question

> Can a lightweight neural network detect abnormal sensor behaviour while using significantly fewer computational resources than a conventional ML model?

## Engineering Problem

Robotic systems and embedded devices depend on sensor measurements for perception, localisation, and control. Sensors can exhibit abnormal behaviour due to noise, corruption, drift, or hardware faults. Detecting these anomalies before they propagate through the system can improve robustness, particularly when computation, memory, and power are constrained.

In resource-constrained environments, the choice of ML model involves trade-offs between detection performance and computational cost. This project evaluates whether a manually implemented neural network built from first principles using NumPy can provide effective anomaly detection while remaining lightweight enough for embedded deployment.

## Why This Matters

A robot that cannot distinguish between valid sensor readings and anomalies may make incorrect control decisions, leading to unsafe behaviour or reduced task performance. Conventional ML approaches often require substantial computational resources that may not be available on embedded processors or microcontrollers.

By implementing a neural network from scratch without deep learning frameworks, we can:

- Understand the computational requirements at the implementation level
- Measure exact parameter counts, memory usage, and inference latency
- Evaluate trade-offs between model complexity and performance
- Determine if a lightweight approach is viable for real-time embedded applications

## Approach

This project implements a fully connected neural network from first principles using only NumPy. The implementation includes:

- Forward propagation with ReLU hidden activations and softmax output
- Cross-entropy loss calculation
- Backpropagation using the chain rule
- Gradient descent parameter updates
- He (Kaiming) weight initialization

The network is evaluated on two experiments:

1. **Neural Network Fundamentals**: A two-moons classification task validates that the implementation correctly learns non-linear decision boundaries
2. **Sensor Anomaly Detection**: The network distinguishes normal sensor behaviour from anomalous patterns including spikes, drift, abnormal ranges, and high noise

## Experiment 1: Neural Network Fundamentals

### Objective

Validate that the manually implemented neural network can correctly perform:

- Forward propagation
- Activation functions
- Loss calculation
- Backpropagation
- Gradient-based optimisation
- Parameter updates
- Classification

This experiment answers the question: **Can the neural network implementation itself learn a nonlinear classification problem correctly?**

### Two-Moons Classification

The two-moons dataset consists of interleaved crescent-shaped clusters in 2D that are not linearly separable. This provides a clean validation that the network can learn curved decision boundaries.

**Setup**

| Setting | Value |
|---------|-------|
| Samples | 1,000 (500 per class) |
| Noise | Gaussian, σ = 0.15 |
| Train / test split | 80% / 20% (800 train, 200 test) |
| Preprocessing | Feature standardization (zero mean, unit variance) |
| Architecture | `[2, 32, 16, 2]` |
| Epochs | 1,500 |
| Learning rate | 0.2 |
| Optimizer | Full-batch gradient descent |

**Results**

| Split | Accuracy |
|-------|----------|
| Train | 99.38% |
| Test | **99.50%** |

The network achieves 99.50% test accuracy, confirming that forward propagation, softmax + cross-entropy backpropagation, and ReLU gradients are all implemented correctly. The minimal gap between train and test accuracy indicates no significant overfitting.

## Experiment 2: Sensor Anomaly Detection

### Objective

Evaluate whether the neural network can distinguish normal sensor behaviour from abnormal/anomalous patterns in a realistic but appropriately scoped setup.

### Sensor Data Generation

Synthetic sensor time-series data simulates a periodic signal (e.g., a sensor reading following a sinusoidal pattern) with the following characteristics:

**Normal behaviour**: Sine wave with small Gaussian noise (σ = 0.05)

**Anomalous behaviour** (four types):
- **Sudden spikes**: Brief large-amplitude deviations
- **Sensor drift**: Gradual offset increase over time
- **Abnormal range**: Constant offset outside normal operating range
- **High noise**: Elevated noise levels

Each sample consists of 50 time-steps (window size = 50), providing sufficient temporal context for pattern recognition.

### Baseline Comparison

The neural network is compared against logistic regression, a lightweight conventional ML baseline that serves as a reasonable point of comparison for resource-constrained applications.

**Setup**

| Setting | Value |
|---------|-------|
| Total samples | 2,000 (1,000 per class) |
| Train / test split | 80% / 20% (1,600 train, 400 test) |
| Input features | 50 (time-steps) |
| Preprocessing | Feature standardization |
| Neural network architecture | `[50, 32, 16, 2]` |
| Neural network epochs | 200 |
| Neural network learning rate | 0.1 |
| Logistic regression epochs | 1,000 |
| Logistic regression learning rate | 0.1 |

**Performance Comparison**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8750 | 1.0000 | 0.7573 | 0.8619 |
| NumPy Neural Network | 0.9925 | 1.0000 | 0.9854 | 0.9927 |

The neural network achieves 99.25% accuracy compared to 87.50% for logistic regression, representing a 13.4% absolute improvement. Both models achieve perfect precision (no false positives), but the neural network significantly improves recall (98.54% vs 75.73%), meaning it misses far fewer actual anomalies.

**Computational Efficiency**

| Model | Parameters | Model Size (KB) | Training Time (s) | Inference Time (ms) |
|---|---:|---:|---:|---:|
| Logistic Regression | 51 | 0.40 | 0.098 | 0.015 |
| NumPy Neural Network | 2,194 | 17.14 | 0.417 | 0.215 |

The neural network uses 43× more parameters (2,194 vs 51) and 43× more memory (17.14 KB vs 0.40 KB) than logistic regression. Training time is 4.3× longer (0.417s vs 0.098s), and inference latency is 14× higher (0.215ms vs 0.015ms).

**Trade-off Analysis**

The neural network provides substantially better anomaly detection performance (+13.4% accuracy, +22.8% recall) at the cost of increased computational resources. For embedded applications where detection accuracy is critical and the computational budget permits a 0.2ms inference time and ~17 KB memory footprint, the neural network may be justified. For extremely constrained environments, logistic regression provides a viable baseline with minimal resource usage.

## Robustness Under Sensor Noise

### Objective

Evaluate how anomaly detection performance degrades as sensor noise increases, simulating real-world conditions where sensor quality varies or environmental interference is present.

### Methodology

The neural network is trained and evaluated at five controlled noise levels:

- 0%: No additional noise (baseline)
- 5%: Low noise
- 10%: Moderate noise
- 20%: High noise
- 30%: Very high noise

Noise is added as Gaussian variation to the sensor signal.

**Results**

| Noise Level | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0% | 0.9925 | 1.0000 | 0.9854 | 0.9927 |
| 5% | 0.9925 | 1.0000 | 0.9854 | 0.9927 |
| 10% | 0.9925 | 1.0000 | 0.9854 | 0.9927 |
| 20% | 0.9875 | 0.9951 | 0.9806 | 0.9878 |
| 30% | 0.9800 | 0.9901 | 0.9709 | 0.9804 |

**Interpretation**

The network maintains near-perfect performance up to 10% noise, with no degradation in accuracy or recall. At 20% noise, accuracy drops by 0.5 percentage points (99.25% → 98.75%). At 30% noise, accuracy drops by 1.25 percentage points (99.25% → 98.00%). This gradual degradation suggests the network is reasonably robust to moderate sensor noise but would benefit from additional regularization or training with noise augmentation for very noisy environments.

## Computational Efficiency

### Objective

Quantify the computational requirements of the neural network to assess suitability for embedded deployment.

### Measured Metrics

For the medium architecture `[50, 32, 16, 2]`:

| Metric | Value |
|--------|-------|
| Trainable parameters | 2,194 |
| Model size | 17.14 KB |
| Training time | 0.417 s |
| Inference latency | 0.215 ms |
| Parameters per input feature | 43.9 |

### Why These Metrics Matter

- **Parameter count**: Determines memory footprint and affects cache utilisation
- **Model size**: Directly impacts storage requirements on embedded devices
- **Inference latency**: Critical for real-time control loops where decisions must be made within strict timing constraints
- **Training time**: Relevant for on-device learning or adaptation scenarios

A model is not necessarily better simply because it achieves higher predictive performance. For an embedded robotic system, accuracy must be considered alongside latency, memory footprint, and computational cost.

## Architecture Trade-offs

### Objective

Evaluate how model complexity affects performance and computational cost to inform design decisions for resource-constrained deployment.

### Methodology

Three network architectures are compared:

- **Small**: `[50, 8, 2]` : Minimal hidden capacity
- **Medium**: `[50, 32, 16, 2]` : Balanced capacity (used in main experiments)
- **Large**: `[50, 64, 32, 16, 2]` : High capacity

**Results**

| Architecture | Parameters | Accuracy | Inference Latency (ms) | Model Size (KB) |
|---|---:|---:|---:|---:|
| Small | 426 | 0.9350 | 0.042 | 3.33 |
| Medium | 2,194 | 0.9950 | 0.214 | 17.14 |
| Large | 5,906 | 1.0000 | 0.236 | 46.14 |

**Interpretation**

- **Small architecture**: Achieves 93.5% accuracy with only 426 parameters and 0.042ms inference latency. Suitable for extremely constrained environments where moderate detection performance is acceptable.
- **Medium architecture**: Achieves 99.5% accuracy with 2,194 parameters and 0.214ms inference latency. Provides the best balance of performance and computational cost for most embedded applications.
- **Large architecture**: Achieves perfect accuracy (100%) with 5,906 parameters and 0.236ms inference latency. Minimal latency increase over medium architecture, but 2.7× more parameters. Only justified if perfect detection is required and memory is available.

The small-to-medium architecture transition provides the largest accuracy gain (+6.0%) for a reasonable increase in resources. The medium-to-large transition provides diminishing returns (+0.5% accuracy) for a substantial increase in parameters.

## Ablation Study

### Objective

Evaluate which design decisions materially affect performance.

### Learning Rate

| Learning Rate | Accuracy | F1 |
|---:|---:|---:|
| 0.01 | 0.9375 | 0.9354 |
| 0.1 | 0.9950 | 0.9951 |
| 0.5 | 0.9950 | 0.9951 |

A learning rate of 0.01 results in significantly poorer performance (93.75% vs 99.50%), suggesting underfitting due to slow convergence. Learning rates of 0.1 and 0.5 achieve equivalent performance, indicating that the model is robust to moderate learning rate variation within this range.

### Feature Standardization

| Configuration | Accuracy | F1 |
|---|---:|---:|
| With standardization | 0.9850 | 0.9852 |
| Without standardization | 0.9925 | 0.9927 |

Counter-intuitively, the model performs slightly better without feature standardization (99.25% vs 98.50%). This may be due to the synthetic sensor data already having a consistent scale (sine wave outputs in [-1, 1] range) and standardization potentially removing useful magnitude information relevant to anomaly detection.

## Results Summary

### Fundamentals

The two-moons experiment confirms the neural network implementation is correct, achieving 99.50% test accuracy on a non-linear classification problem.

### Sensor Anomaly Detection

The neural network achieves 99.25% accuracy on sensor anomaly detection, compared to 87.50% for logistic regression. The improvement comes primarily from better recall (98.54% vs 75.73%), meaning the neural network misses far fewer actual anomalies.

### Robustness

The network maintains >98% accuracy even at 30% noise levels, demonstrating reasonable robustness to sensor noise. Performance degradation is gradual rather than catastrophic.

### Computational Efficiency

The medium architecture requires 2,194 parameters (17.14 KB), trains in 0.417s, and infers in 0.215ms. This is within the computational budget of many embedded processors, though significantly more expensive than logistic regression (51 parameters, 0.40 KB, 0.015ms inference).

### Architecture Trade-offs

The medium architecture provides the best balance of performance (99.5% accuracy) and computational cost for most embedded applications. The small architecture is suitable for extreme constraints with moderate performance requirements. The large architecture offers diminishing returns.

## Engineering Trade-offs

The central question was: **Can a lightweight neural network detect abnormal sensor behaviour while using significantly fewer computational resources than a conventional ML model?**

Based on the measured results:

**Detection Performance**: The neural network significantly outperforms logistic regression in anomaly detection (99.25% vs 87.50% accuracy, +22.8% recall). This is a substantial improvement that could translate to fewer missed anomalies in a real robotic system.

**Computational Cost**: The neural network requires significantly more computational resources than logistic regression:
- 43× more parameters (2,194 vs 51)
- 43× more memory (17.14 KB vs 0.40 KB)
- 4.3× longer training time (0.417s vs 0.098s)
- 14× higher inference latency (0.215ms vs 0.015ms)

**Conclusion**: The neural network does **not** use fewer computational resources than the conventional baseline. However, it provides substantially better anomaly detection performance. The engineering trade-off is clear:

- Use logistic regression if computational resources are extremely constrained and moderate detection performance (87.5% accuracy) is acceptable
- Use the neural network if the computational budget permits a 0.2ms inference time and ~17 KB memory footprint, and the improved detection performance (99.25% accuracy) is valuable for the application

For many embedded robotic applications where missed anomalies could lead to unsafe behaviour, the 0.2ms inference time and 17 KB memory footprint may be acceptable given the significant improvement in detection capability. The small architecture variant (426 parameters, 0.042ms inference, 93.5% accuracy) provides an intermediate option for more constrained environments.

## Limitations

1. **Synthetic data**: Experiments use synthetic sensor data. Real sensor logs may exhibit different anomaly patterns and noise characteristics.
2. **No physical hardware**: Results are from simulation only. Performance on actual embedded processors may vary due to memory hierarchy, instruction sets, and implementation details.
3. **Limited anomaly types**: Four anomaly types were evaluated. Real-world sensors may exhibit additional failure modes.
4. **Batch processing**: Current implementation processes batches. Streaming/online inference would require additional design considerations.
5. **No temporal modelling**: The network treats each time window independently. Recurrent or temporal architectures could potentially capture time-dependencies in sensor behaviour.
6. **No comparison to embedded-optimised models**: Comparison is against logistic regression. Specialised embedded ML models (e.g., quantised networks, TinyML approaches) were not evaluated.

## Future Work

- **Real sensor logs**: Test with actual sensor data from robotic platforms
- **Physical hardware**: Deploy to embedded processors (e.g., ARM Cortex-M, microcontrollers) and measure real-world performance
- **Additional anomaly types**: Evaluate sensor failures, correlated faults, and multi-sensor anomalies
- **Online/streaming detection**: Implement streaming inference for real-time monitoring
- **Quantisation**: Evaluate fixed-point and low-precision quantisation to reduce memory footprint
- **Temporal modelling**: Explore recurrent architectures or sliding-window approaches for time-dependent patterns
- **Additional baselines**: Compare against other lightweight ML methods (e.g., decision trees, SVMs)
- **Failure mode analysis**: Evaluate performance under sensor complete failure and extreme corruption

## Repository Structure

```
numpy-neural-network/
├── neural_network.py              # Core neural network implementation
├── logistic_regression.py         # Baseline logistic regression implementation
├── train.py                       # Original two-moons and MNIST demos
├── sensor_anomaly_experiment.py   # Sensor anomaly detection experiment
├── comprehensive_experiments.py  # All engineering experiments
├── visualize.py                    # Visualization utilities
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.24.0
- matplotlib>=3.7.0 (for visualizations)

## Usage

### Run Fundamentals Experiment (Two Moons + MNIST)

```bash
python train.py
```

### Run Sensor Anomaly Detection Only

```bash
python sensor_anomaly_experiment.py
```

### Run All Engineering Experiments

```bash
python comprehensive_experiments.py
```

This runs:
- Baseline comparison (neural network vs logistic regression)
- Robustness experiment (varying noise levels)
- Architecture comparison (small/medium/large)
- Ablation study (learning rates, standardization)

### Using the Neural Network Class

```python
from neural_network import NeuralNetwork

# Define architecture
model = NeuralNetwork([input_dim, hidden_dim1, hidden_dim2, num_classes])

# Train
history = model.fit(X_train, y_train, epochs=200, learning_rate=0.1)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

`X` can be shaped as `(num_samples, num_features)` or `(num_features, num_samples)` (the class normalizes internally). Labels should be integer class indices.

## Implementation Details

### Forward Propagation

For a batch of `m` samples stored as columns:

| Step | Formula |
|------|---------|
| Linear layer | `Z^[l] = W^[l] A^[l-1] + b^[l]` |
| ReLU | `A^[l] = max(0, Z^[l])` |
| Softmax | `A^[L] = exp(Z^[L] - max) / sum(exp(...))` |
| Loss | `L = -1/m sum(y log A^[L])` |

### Backpropagation

| Step | Formula |
|------|---------|
| Output gradient | `dZ^[L] = A^[L] - Y` |
| Hidden gradient | `dZ^[l] = (W^[l+1]^T dZ^[l+1]) ⊙ ReLU'(Z^[l])` |
| Weight gradient | `dW^[l] = 1/m dZ^[l] (A^[l-1])^T` |
| Bias gradient | `db^[l] = 1/m sum(dZ^[l])` |

Softmax subtracts the per-column maximum before exponentiating to avoid overflow. Cross-entropy clips probabilities slightly so `log(0)` never appears.

The output-layer gradient simplifies to `A - Y` when softmax and cross-entropy are combined, which is mathematically cleaner and more numerically stable than computing them separately.

### Parameter Initialization

Weights are initialized using He (Kaiming) initialization: `W ~ N(0, sqrt(2/fan_in))`. This keeps early activations in a sensible range for ReLU networks. Biases are initialized to zero.

## License

This project is provided as-is for educational and engineering evaluation purposes.
