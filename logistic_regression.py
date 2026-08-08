import numpy as np
import time


class LogisticRegression:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.zeros((input_dim, 1))
        self.bias = np.zeros((1, 1))
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01, verbose=True, print_every=100):
        features = X.T if X.shape[0] != self.input_dim else X
        labels = y.reshape(1, -1)
        num_samples = features.shape[1]
        
        for epoch in range(1, epochs + 1):
            z = self.weights.T @ features + self.bias
            predictions = self._sigmoid(z)
            
            epsilon = 1e-12
            predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
            loss = -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
            
            pred_labels = (predictions >= 0.5).astype(int)
            accuracy = np.mean(pred_labels == labels)
            
            dz = predictions - labels
            dw = (features @ dz.T) / num_samples
            db = np.mean(dz)
            
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            if verbose and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
                print(f"Epoch {epoch:4d} | loss: {loss:.6f} | accuracy: {accuracy * 100:.2f}%")
    
    def predict(self, X):
        features = X.T if X.shape[0] != self.input_dim else X
        z = self.weights.T @ features + self.bias
        predictions = self._sigmoid(z)
        return (predictions >= 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        features = X.T if X.shape[0] != self.input_dim else X
        z = self.weights.T @ features + self.bias
        probabilities = self._sigmoid(z)
        proba = np.vstack([1 - probabilities, probabilities]).T
        return proba
    
    def count_parameters(self):
        return self.weights.size + self.bias.size


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


def measure_inference_time(model, X, num_runs=100):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(X)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)
