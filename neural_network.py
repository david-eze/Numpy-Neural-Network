import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        if len(layer_sizes) < 2:
            raise ValueError("Architecture needs at least an input and output layer.")

        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes) - 1

        self.weights = []
        self.biases = []

        for layer_index in range(self.num_layers):
            fan_in = layer_sizes[layer_index]
            fan_out = layer_sizes[layer_index + 1]
            scale = np.sqrt(2.0 / fan_in)
            weight_matrix = np.random.randn(fan_out, fan_in) * scale
            bias_vector = np.zeros((fan_out, 1))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

        self.cache = {}

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def _softmax(z):
        shifted = z - np.max(z, axis=0, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=0, keepdims=True)

    @staticmethod
    def _one_hot(labels, num_classes):
        labels = np.asarray(labels).reshape(1, -1)
        encoded = np.zeros((num_classes, labels.shape[1]))
        encoded[labels, np.arange(labels.shape[1])] = 1.0
        return encoded

    @staticmethod
    def _cross_entropy_loss(probabilities, targets):
        epsilon = 1e-12
        clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        sample_losses = -np.sum(targets * np.log(clipped), axis=0)
        return float(np.mean(sample_losses))

    def _forward(self, features):
        activations = [features]
        pre_activations = []

        current_activation = features
        for layer_index in range(self.num_layers):
            z = self.weights[layer_index] @ current_activation + self.biases[layer_index]
            pre_activations.append(z)

            if layer_index == self.num_layers - 1:
                current_activation = self._softmax(z)
            else:
                current_activation = self._relu(z)

            activations.append(current_activation)

        self.cache = {
            "activations": activations,
            "pre_activations": pre_activations,
        }
        return activations[-1]

    def _backward(self, targets):
        activations = self.cache["activations"]
        pre_activations = self.cache["pre_activations"]
        batch_size = targets.shape[1]

        weight_gradients = [None] * self.num_layers
        bias_gradients = [None] * self.num_layers

        delta = activations[-1] - targets

        for layer_index in reversed(range(self.num_layers)):
            previous_activation = activations[layer_index]
            weight_gradients[layer_index] = (delta @ previous_activation.T) / batch_size
            bias_gradients[layer_index] = np.sum(delta, axis=1, keepdims=True) / batch_size

            if layer_index > 0:
                delta = self.weights[layer_index].T @ delta
                delta = delta * self._relu_derivative(pre_activations[layer_index - 1])

        return weight_gradients, bias_gradients

    def _update_parameters(self, weight_gradients, bias_gradients, learning_rate):
        for layer_index in range(self.num_layers):
            self.weights[layer_index] -= learning_rate * weight_gradients[layer_index]
            self.biases[layer_index] -= learning_rate * bias_gradients[layer_index]

    def fit(self, X, y, epochs=1000, learning_rate=0.1, verbose=True, print_every=100):
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.shape[0] != self.layer_sizes[0]:
            features = features.T

        num_classes = self.layer_sizes[-1]
        targets = self._one_hot(y, num_classes)
        history = {"loss": [], "accuracy": []}

        for epoch in range(1, epochs + 1):
            probabilities = self._forward(features)
            loss = self._cross_entropy_loss(probabilities, targets)
            predictions = np.argmax(probabilities, axis=0)
            accuracy = float(np.mean(predictions == np.asarray(y).reshape(-1)))

            weight_gradients, bias_gradients = self._backward(targets)
            self._update_parameters(weight_gradients, bias_gradients, learning_rate)

            history["loss"].append(loss)
            history["accuracy"].append(accuracy)

            if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
                print(f"Epoch {epoch:4d} | loss: {loss:.6f} | accuracy: {accuracy * 100:.2f}%")

        return history

    def predict(self, X):
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.shape[0] != self.layer_sizes[0]:
            features = features.T

        probabilities = self._forward(features)
        return np.argmax(probabilities, axis=0)

    def predict_proba(self, X):
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.shape[0] != self.layer_sizes[0]:
            features = features.T

        return self._forward(features)
