import os
from pathlib import Path

import numpy as np

from neural_network import NeuralNetwork
from train import (
    download_mnist,
    generate_moon_data,
    standardize_features,
    train_test_split,
)


OUTPUT_DIR = Path(__file__).parent / "visualizations"

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_curves(history, title, filename):
    """Save loss and accuracy over the training run."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    figure.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["loss"], color="#2463a6", linewidth=2)
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, np.asarray(history["accuracy"]) * 100, color="#16835c", linewidth=2)
    axes[1].set(title="Training accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[1].grid(alpha=0.25)
    figure.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_moon_decision_boundary(model, X, y):
    """Show the learned non-linear classification boundary for two moons."""
    padding = 0.5
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350)
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    probabilities = model.predict_proba(grid)[1].reshape(grid_x.shape)

    figure, axis = plt.subplots(figsize=(7.2, 5.7), layout="constrained")
    contour = axis.contourf(grid_x, grid_y, probabilities, levels=30, cmap="RdBu", alpha=0.72)
    axis.contour(grid_x, grid_y, probabilities, levels=[0.5], colors="black", linewidths=1.5)
    scatter = axis.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="white", linewidth=0.35, s=22)
    figure.colorbar(contour, ax=axis, label="Predicted probability of class 1")
    axis.set(
        title="Two moons: learned decision boundary",
        xlabel="Standardized feature 1",
        ylabel="Standardized feature 2",
    )
    axis.legend(*scatter.legend_elements(), title="True class", loc="upper right")
    figure.savefig(OUTPUT_DIR / "two_moons_decision_boundary.png", dpi=180)
    plt.close(figure)


def plot_mnist_predictions(images, labels, predictions):
    """Save a grid of test digits, with errors shown in red."""
    figure, axes = plt.subplots(4, 5, figsize=(8, 7.2), layout="constrained")
    figure.suptitle("MNIST test predictions (red = incorrect)", fontsize=14, fontweight="bold")
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index].reshape(28, 28), cmap="gray_r")
        correct = predictions[index] == labels[index]
        axis.set_title(
            f"true {labels[index]}  /  predicted {predictions[index]}",
            color="#16835c" if correct else "#c63c3c", fontsize=9,
        )
        axis.axis("off")
    figure.savefig(OUTPUT_DIR / "mnist_predictions.png", dpi=180)
    plt.close(figure)


def plot_confusion_matrix(labels, predictions):
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    figure, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Number of test images")
    axis.set(
        title="MNIST confusion matrix",
        xlabel="Predicted digit",
        ylabel="True digit",
        xticks=range(10),
        yticks=range(10),
    )
    for row in range(10):
        for column in range(10):
            colour = "white" if matrix[row, column] > matrix.max() * 0.5 else "#1e3c5a"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=colour, fontsize=8)
    figure.savefig(OUTPUT_DIR / "mnist_confusion_matrix.png", dpi=180)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(42)

    X, y = generate_moon_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = standardize_features(X_train, X_test)
    moon_model = NeuralNetwork([2, 32, 16, 2])
    moon_history = moon_model.fit(X_train, y_train, epochs=1500, learning_rate=0.2, verbose=False)
    save_training_curves(moon_history, "Two-moons training", "two_moons_training_curves.png")
    plot_moon_decision_boundary(moon_model, X_test, y_test)

    train_images, train_labels, test_images, test_labels = download_mnist()
    train_images, train_labels = train_images[:5000], train_labels[:5000]
    test_images, test_labels = test_images[:1000], test_labels[:1000]
    train_images, normalized_test_images = standardize_features(train_images, test_images)
    mnist_model = NeuralNetwork([784, 128, 64, 10])
    mnist_history = mnist_model.fit(train_images, train_labels, epochs=30, learning_rate=0.5, verbose=False)
    predictions = mnist_model.predict(normalized_test_images)
    save_training_curves(mnist_history, "MNIST training", "mnist_training_curves.png")
    plot_mnist_predictions(test_images, test_labels, predictions)
    plot_confusion_matrix(test_labels, predictions)
    print(f"Saved visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
