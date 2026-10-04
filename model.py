import numpy as np
import pandas as pd

# ============================
# LOAD DATA
# ============================

df = pd.read_csv("data_scaled.csv")
X = df[["F1", "F2"]].values
y = df["label"].values.astype(int)

# One-hot encoding
def one_hot(y, n_classes=3):
    return np.eye(n_classes)[y]

y_onehot = one_hot(y)

# Train/Test split
np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y_onehot[idx[:split]], y_onehot[idx[split:]]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================
# MODEL
# ============================

class NeuralNetwork:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr
        self.W = []
        self.b = []
        np.random.seed(42)
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * 0.5
            b = np.zeros((1, layers[i + 1]))
            self.W.append(w)
            self.b.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.zs = []
        self.acts = [X]
        cur = X
        for i in range(len(self.W) - 1):
            z = cur @ self.W[i] + self.b[i]
            self.zs.append(z)
            cur = self.relu(z)
            self.acts.append(cur)
        z_out = cur @ self.W[-1] + self.b[-1]
        self.zs.append(z_out)
        out = self.softmax(z_out)
        self.acts.append(out)
        return out

    def loss(self, y_pred, y_true):
        eps = 1e-9
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    def backward(self, y_true):
        m = y_true.shape[0]
        delta = (self.acts[-1] - y_true) / m
        for i in range(len(self.W) - 1, -1, -1):
            dW = self.acts[i].T @ delta
            db = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.relu_deriv(self.zs[i - 1])
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def fit(self, X, y, epochs=2000, batch_size=32):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]
            for i in range(0, n, batch_size):
                Xb = X_s[i:i + batch_size]
                yb = y_s[i:i + batch_size]
                self.forward(Xb)
                self.backward(yb)
            if epoch % 200 == 0:
                out = self.forward(X)
                print(f"Epoch {epoch}: Loss = {self.loss(out, y):.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ============================
# TRAIN
# ============================

model = NeuralNetwork(layers=[2, 4, 4, 4, 3], lr=0.01)
model.fit(X_train, y_train, epochs=2000, batch_size=32)

# ============================
# EVALUATE
# ============================

pred = model.predict(X_test)
true = np.argmax(y_test, axis=1)
acc = np.mean(pred == true)

print(f"\nTest Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(true, pred):
    cm[t][p] += 1
print(cm)
