import numpy as np
import time
from neural_network import NeuralNetwork


def generate_sensor_data(num_samples=2000, window_size=50, random_seed=42):
    rng = np.random.default_rng(random_seed)
    
    X = []
    y = []
    
    samples_per_class = num_samples // 2
    
    for i in range(samples_per_class):
        t = np.linspace(0, 2 * np.pi, window_size)
        base_signal = np.sin(t) + rng.normal(0, 0.05, window_size)
        X.append(base_signal)
        y.append(0)
    
    for i in range(samples_per_class):
        t = np.linspace(0, 2 * np.pi, window_size)
        anomaly_type = rng.integers(0, 4)
        
        if anomaly_type == 0:
            signal = np.sin(t) + rng.normal(0, 0.05, window_size)
            spike_idx = rng.integers(5, window_size - 5)
            signal[spike_idx:spike_idx+3] += rng.uniform(2.0, 3.0, 3)
        elif anomaly_type == 1:
            drift = np.linspace(0, 1.5, window_size)
            signal = np.sin(t) + drift + rng.normal(0, 0.05, window_size)
        elif anomaly_type == 2:
            signal = np.sin(t) + 2.0 + rng.normal(0, 0.05, window_size)
        else:
            signal = np.sin(t) + rng.normal(0, 0.4, window_size)
        
        X.append(signal)
        y.append(1)
    
    X = np.array(X)
    y = np.array(y, dtype=int)
    
    indices = rng.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    return X, y


def standardize_features(train_features, test_features):
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_features - mean) / std, (test_features - mean) / std


def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1


def count_parameters(model):
    total = 0
    for w, b in zip(model.weights, model.biases):
        total += w.size + b.size
    return total


def measure_inference_time(model, X, num_runs=100):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(X)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)


def run_sensor_anomaly_experiment():
    print("=" * 60)
    print("Sensor Anomaly Detection Experiment")
    print("=" * 60)
    
    np.random.seed(42)
    
    X, y = generate_sensor_data(num_samples=2000, window_size=50)
    
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train_std, X_test_std = standardize_features(X_train, X_test)
    
    print("\nTraining NumPy Neural Network...")
    model = NeuralNetwork([50, 32, 16, 2])
    
    start_train = time.perf_counter()
    model.fit(X_train_std, y_train, epochs=200, learning_rate=0.1, 
              verbose=True, print_every=50)
    train_time = time.perf_counter() - start_train
    
    train_pred = model.predict(X_train_std)
    test_pred = model.predict(X_test_std)
    
    train_acc, train_prec, train_rec, train_f1 = calculate_metrics(y_train, train_pred)
    test_acc, test_prec, test_rec, test_f1 = calculate_metrics(y_test, test_pred)
    
    num_params = count_parameters(model)
    inference_time = measure_inference_time(model, X_test_std)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Training time: {train_time:.3f}s")
    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy:  {test_acc * 100:.2f}%")
    print(f"Test precision: {test_prec:.4f}")
    print(f"Test recall:    {test_rec:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Parameters:     {num_params}")
    print(f"Avg inference time: {inference_time * 1000:.3f}ms")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'parameters': num_params,
        'training_time': train_time,
        'inference_time': inference_time
    }


if __name__ == "__main__":
    results = run_sensor_anomaly_experiment()
