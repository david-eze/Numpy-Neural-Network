import numpy as np
import time
from neural_network import NeuralNetwork
from logistic_regression import LogisticRegression


def generate_sensor_data(num_samples=2000, window_size=50, noise_level=0.05, random_seed=42):
    rng = np.random.default_rng(random_seed)
    
    X = []
    y = []
    
    samples_per_class = num_samples // 2
    
    for i in range(samples_per_class):
        t = np.linspace(0, 2 * np.pi, window_size)
        base_signal = np.sin(t) + rng.normal(0, noise_level, window_size)
        X.append(base_signal)
        y.append(0)
    
    for i in range(samples_per_class):
        t = np.linspace(0, 2 * np.pi, window_size)
        anomaly_type = rng.integers(0, 4)
        
        if anomaly_type == 0:
            signal = np.sin(t) + rng.normal(0, noise_level, window_size)
            spike_idx = rng.integers(5, window_size - 5)
            signal[spike_idx:spike_idx+3] += rng.uniform(2.0, 3.0, 3)
        elif anomaly_type == 1:
            drift = np.linspace(0, 1.5, window_size)
            signal = np.sin(t) + drift + rng.normal(0, noise_level, window_size)
        elif anomaly_type == 2:
            signal = np.sin(t) + 2.0 + rng.normal(0, noise_level, window_size)
        else:
            signal = np.sin(t) + rng.normal(0, 0.4 + noise_level, window_size)
        
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
    if isinstance(model, LogisticRegression):
        return model.count_parameters()
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


def estimate_model_size(model):
    param_count = count_parameters(model)
    return param_count * 8


def run_baseline_comparison():
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON: Neural Network vs Logistic Regression")
    print("=" * 60)
    
    np.random.seed(42)
    
    X, y = generate_sensor_data(num_samples=2000, window_size=50)
    
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train_std, X_test_std = standardize_features(X_train, X_test)
    
    results = {}
    
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(input_dim=50)
    
    start_train = time.perf_counter()
    lr_model.fit(X_train_std, y_train, epochs=1000, learning_rate=0.1, 
                 verbose=True, print_every=200)
    lr_train_time = time.perf_counter() - start_train
    
    lr_test_pred = lr_model.predict(X_test_std)
    lr_acc, lr_prec, lr_rec, lr_f1 = calculate_metrics(y_test, lr_test_pred)
    lr_params = count_parameters(lr_model)
    lr_inference_time = measure_inference_time(lr_model, X_test_std)
    lr_model_size = estimate_model_size(lr_model)
    
    results['logistic_regression'] = {
        'accuracy': lr_acc,
        'precision': lr_prec,
        'recall': lr_rec,
        'f1': lr_f1,
        'parameters': lr_params,
        'model_size': lr_model_size,
        'training_time': lr_train_time,
        'inference_time': lr_inference_time
    }
    
    print("\nTraining Neural Network...")
    nn_model = NeuralNetwork([50, 32, 16, 2])
    
    start_train = time.perf_counter()
    nn_model.fit(X_train_std, y_train, epochs=200, learning_rate=0.1, 
                 verbose=True, print_every=50)
    nn_train_time = time.perf_counter() - start_train
    
    nn_test_pred = nn_model.predict(X_test_std)
    nn_acc, nn_prec, nn_rec, nn_f1 = calculate_metrics(y_test, nn_test_pred)
    nn_params = count_parameters(nn_model)
    nn_inference_time = measure_inference_time(nn_model, X_test_std)
    nn_model_size = estimate_model_size(nn_model)
    
    results['neural_network'] = {
        'accuracy': nn_acc,
        'precision': nn_prec,
        'recall': nn_rec,
        'f1': nn_f1,
        'parameters': nn_params,
        'model_size': nn_model_size,
        'training_time': nn_train_time,
        'inference_time': nn_inference_time
    }
    
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    print(f"{'Logistic Regression':<25} {lr_acc:>10.4f} {lr_prec:>10.4f} {lr_rec:>10.4f} {lr_f1:>10.4f}")
    print(f"{'NumPy Neural Network':<25} {nn_acc:>10.4f} {nn_prec:>10.4f} {nn_rec:>10.4f} {nn_f1:>10.4f}")
    
    print("\n" + "=" * 60)
    print("COMPUTATIONAL EFFICIENCY")
    print("=" * 60)
    print(f"{'Model':<25} {'Params':>10} {'Size (KB)':>12} {'Train (s)':>10} {'Infer (ms)':>12}")
    print("-" * 60)
    print(f"{'Logistic Regression':<25} {lr_params:>10} {lr_model_size/1024:>12.2f} {lr_train_time:>10.3f} {lr_inference_time*1000:>12.3f}")
    print(f"{'NumPy Neural Network':<25} {nn_params:>10} {nn_model_size/1024:>12.2f} {nn_train_time:>10.3f} {nn_inference_time*1000:>12.3f}")
    
    return results


def run_robustness_experiment():
    print("\n" + "=" * 60)
    print("ROBUSTNESS EXPERIMENT: Performance vs Sensor Noise")
    print("=" * 60)
    
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    results = []
    
    for noise in noise_levels:
        print(f"\nTesting with noise level: {noise*100:.0f}%")
        np.random.seed(42)
        
        X, y = generate_sensor_data(num_samples=2000, window_size=50, noise_level=noise)
        
        split_idx = int(0.8 * len(y))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_train_std, X_test_std = standardize_features(X_train, X_test)
        
        model = NeuralNetwork([50, 32, 16, 2])
        model.fit(X_train_std, y_train, epochs=200, learning_rate=0.1, verbose=False)
        
        test_pred = model.predict(X_test_std)
        acc, prec, rec, f1 = calculate_metrics(y_test, test_pred)
        
        results.append({
            'noise_level': noise,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    
    print("\n" + "=" * 60)
    print("ROBUSTNESS RESULTS")
    print("=" * 60)
    print(f"{'Noise':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['noise_level']*100:>6.0f}%{'':>3} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")
    
    return results


def run_architecture_comparison():
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON: Model Size vs Performance")
    print("=" * 60)
    
    architectures = [
        [50, 8, 2],
        [50, 32, 16, 2],
        [50, 64, 32, 16, 2]
    ]
    
    arch_names = ['Small', 'Medium', 'Large']
    results = []
    
    np.random.seed(42)
    X, y = generate_sensor_data(num_samples=2000, window_size=50)
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train_std, X_test_std = standardize_features(X_train, X_test)
    
    for arch, name in zip(architectures, arch_names):
        print(f"\nTraining {name} architecture: {arch}")
        
        model = NeuralNetwork(arch)
        model.fit(X_train_std, y_train, epochs=200, learning_rate=0.1, verbose=False)
        
        test_pred = model.predict(X_test_std)
        acc, prec, rec, f1 = calculate_metrics(y_test, test_pred)
        
        params = count_parameters(model)
        inference_time = measure_inference_time(model, X_test_std)
        model_size = estimate_model_size(model)
        
        results.append({
            'architecture': name,
            'layers': arch,
            'parameters': params,
            'accuracy': acc,
            'inference_time': inference_time,
            'model_size': model_size
        })
    
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Architecture':<15} {'Params':>10} {'Accuracy':>10} {'Infer (ms)':>12} {'Size (KB)':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['architecture']:<15} {r['parameters']:>10} {r['accuracy']:>10.4f} {r['inference_time']*1000:>12.3f} {r['model_size']/1024:>12.2f}")
    
    return results


def run_ablation_study():
    print("\n" + "=" * 60)
    print("ABLATION STUDY: Design Choice Impact")
    print("=" * 60)
    
    np.random.seed(42)
    X, y = generate_sensor_data(num_samples=2000, window_size=50)
    split_idx = int(0.8 * len(y))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train_std, X_test_std = standardize_features(X_train, X_test)
    
    results = []
    
    print("\nTesting different learning rates...")
    for lr in [0.01, 0.1, 0.5]:
        model = NeuralNetwork([50, 32, 16, 2])
        model.fit(X_train_std, y_train, epochs=200, learning_rate=lr, verbose=False)
        test_pred = model.predict(X_test_std)
        acc, _, _, f1 = calculate_metrics(y_test, test_pred)
        results.append({'experiment': f'Learning rate {lr}', 'accuracy': acc, 'f1': f1})
    
    print("Testing feature standardization impact...")
    model_std = NeuralNetwork([50, 32, 16, 2])
    model_std.fit(X_train_std, y_train, epochs=200, learning_rate=0.1, verbose=False)
    pred_std = model_std.predict(X_test_std)
    acc_std, _, _, f1_std = calculate_metrics(y_test, pred_std)
    results.append({'experiment': 'With standardization', 'accuracy': acc_std, 'f1': f1_std})
    
    model_no_std = NeuralNetwork([50, 32, 16, 2])
    model_no_std.fit(X_train, y_train, epochs=200, learning_rate=0.1, verbose=False)
    pred_no_std = model_no_std.predict(X_test)
    acc_no_std, _, _, f1_no_std = calculate_metrics(y_test, pred_no_std)
    results.append({'experiment': 'Without standardization', 'accuracy': acc_no_std, 'f1': f1_no_std})
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Experiment':<30} {'Accuracy':>10} {'F1':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['experiment']:<30} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")
    
    return results


def main():
    print("=" * 60)
    print("COMPREHENSIVE ENGINEERING EXPERIMENTS")
    print("NumPy Neural Network for Sensor Anomaly Detection")
    print("=" * 60)
    
    baseline_results = run_baseline_comparison()
    robustness_results = run_robustness_experiment()
    architecture_results = run_architecture_comparison()
    ablation_results = run_ablation_study()
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    
    return {
        'baseline': baseline_results,
        'robustness': robustness_results,
        'architecture': architecture_results,
        'ablation': ablation_results
    }


if __name__ == "__main__":
    results = main()
