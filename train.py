import gzip
import os
import urllib.request

import numpy as np

from neural_network import NeuralNetwork


def generate_moon_data(num_samples=1000, noise=0.15, random_seed=42):
    rng = np.random.default_rng(random_seed)
    samples_per_moon = num_samples // 2

    theta_upper = rng.uniform(0.0, np.pi, samples_per_moon)
    upper_moon = np.column_stack([np.cos(theta_upper), np.sin(theta_upper)])

    theta_lower = rng.uniform(0.0, np.pi, samples_per_moon)
    lower_moon = np.column_stack([1.0 - np.cos(theta_lower), 0.5 - np.sin(theta_lower)])

    X = np.vstack([upper_moon, lower_moon]).astype(float)
    y = np.array([0] * samples_per_moon + [1] * samples_per_moon, dtype=int)
    X += rng.normal(0.0, noise, X.shape)
    return X, y


def train_test_split(X, y, test_ratio=0.2, random_seed=42):
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(y))
    split_index = int(len(y) * (1.0 - test_ratio))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize_features(train_features, test_features):
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_features - mean) / std, (test_features - mean) / std


def download_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    arrays = {}
    for key, filename in files.items():
        local_path = os.path.join(data_dir, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, local_path)

        with gzip.open(local_path, "rb") as file_handle:
            buffer = file_handle.read()

        if "images" in key:
            images = np.frombuffer(buffer, dtype=np.uint8, offset=16)
            arrays[key] = images.reshape(-1, 28 * 28).astype(float) / 255.0
        else:
            arrays[key] = np.frombuffer(buffer, dtype=np.uint8, offset=8).astype(int)

    return arrays["train_images"], arrays["train_labels"], arrays["test_images"], arrays["test_labels"]


def run_moon_demo():
    print("=" * 60)
    print("Two-moons classification demo")
    print("=" * 60)

    X, y = generate_moon_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = standardize_features(X_train, X_test)

    model = NeuralNetwork([2, 32, 16, 2])
    model.fit(X_train, y_train, epochs=1500, learning_rate=0.2, print_every=300)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"\nFinal train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Final test accuracy:  {test_accuracy * 100:.2f}%")


def run_mnist_demo(max_train_samples=5000, max_test_samples=1000):
    print("\n" + "=" * 60)
    print("MNIST digits demo")
    print("=" * 60)

    X_train, y_train, X_test, y_test = download_mnist()
    X_train = X_train[:max_train_samples]
    y_train = y_train[:max_train_samples]
    X_test = X_test[:max_test_samples]
    y_test = y_test[:max_test_samples]

    X_train, X_test = standardize_features(X_train, X_test)

    model = NeuralNetwork([784, 128, 64, 10])
    model.fit(X_train, y_train, epochs=30, learning_rate=0.5, print_every=5)

    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"\nTest accuracy on {len(y_test)} samples: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    np.random.seed(42)
    run_moon_demo()
    run_mnist_demo()
