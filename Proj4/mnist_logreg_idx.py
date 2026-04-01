# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:47:43 2026

@author: zhuofli
"""

import os
import struct
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from softmax import softmax


class SoftmaxLogisticRegression:
    """
    Multiclass logistic regression (softmax regression) from scratch.
    """

    def __init__(self, input_dim, num_classes, lr=0.1, reg_lambda=0.0, seed=42):
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.standard_normal((input_dim, num_classes))
        self.b = np.zeros((1, num_classes))
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.num_classes = num_classes

    def forward(self, X):
        logits = X @ self.W + self.b          # (N, K)
        probs = softmax(logits)               # (N, K)
        return logits, probs

    def compute_loss(self, X, y):
        N = X.shape[0]
        _, probs = self.forward(X)

        eps = 1e-12
        correct_probs = probs[np.arange(N), y]
        ce_loss = -np.mean(np.log(correct_probs + eps))
        reg_loss = 0.5 * self.reg_lambda * np.sum(self.W * self.W)
        return ce_loss + reg_loss

    def compute_gradients(self, X, y):
        N = X.shape[0]
        _, probs = self.forward(X)

        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(N), y] = 1.0

        dlogits = (probs - y_onehot) / N
        dW = X.T @ dlogits + self.reg_lambda * self.W
        db = np.sum(dlogits, axis=0, keepdims=True)
        return dW, db

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=128, verbose=True):
        N = X_train.shape[0]
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        start_time = time.perf_counter()

        for epoch in range(epochs):
            indices = np.random.permutation(N)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for start in range(0, N, batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                dW, db = self.compute_gradients(X_batch, y_batch)
                self.W -= self.lr * dW
                self.b -= self.lr * db

            train_loss = self.compute_loss(X_train, y_train)
            train_acc = self.score(X_train, y_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss = self.compute_loss(X_val, y_val)
                val_acc = self.score(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if verbose:
                    print(
                        f"Epoch {epoch+1:03d}/{epochs} | "
                        f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                        f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch+1:03d}/{epochs} | "
                        f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f}"
                    )

        total_time = time.perf_counter() - start_time
        return history, total_time

    def predict_proba(self, X):
        _, probs = self.forward(X)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def save(self, path):
        np.savez(
            path,
            W=self.W,
            b=self.b,
            lr=self.lr,
            reg_lambda=self.reg_lambda,
            num_classes=self.num_classes
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        input_dim = data["W"].shape[0]
        num_classes = int(data["num_classes"])

        model = cls(
            input_dim=input_dim,
            num_classes=num_classes,
            lr=float(data["lr"]),
            reg_lambda=float(data["reg_lambda"])
        )
        model.W = data["W"]
        model.b = data["b"]
        return model


def load_idx_images(filepath):
    """
    Load MNIST images from idx3-ubyte file.
    Returns:
        images: (N, 28, 28) uint8
    """
    with open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number in image file: {magic}")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_idx_labels(filepath):
    """
    Load MNIST labels from idx1-ubyte file.
    Returns:
        labels: (N,) uint8
    """
    with open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number in label file: {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def preprocess_images(images, normalize=True, flatten=True):
    """
    images: (N, 28, 28)
    return:
        X: (N, 784) float32 if flatten=True
    """
    X = images.astype(np.float32)
    if normalize:
        X /= 255.0
    if flatten:
        X = X.reshape(X.shape[0], -1)
    return X


def split_train_val(X, y, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    N = len(X)
    indices = rng.permutation(N)

    val_size = int(N * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def show_samples(images, labels, num_samples=10):
    plt.figure(figsize=(12, 2.5))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(str(labels[i]))
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def export_predictions_to_excel(y_true, y_pred, out_path):
    """
    Since standard MNIST idx format does not preserve original image filenames,
    we export sample index instead of filename.
    """
    sample_indices = np.arange(len(y_pred))

    df_pred = pd.DataFrame({
        "sample_index": sample_indices,
        "true_label": y_true,
        "predicted_label": y_pred
    })

    label_counts = pd.Series(y_pred).value_counts().sort_index()
    df_count = pd.DataFrame({
        "label": label_counts.index,
        "count": label_counts.values
    })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_pred.to_excel(writer, sheet_name="predictions", index=False)
        df_count.to_excel(writer, sheet_name="label_counts", index=False)

    print(f"Excel output saved to: {out_path}")


def plot_confusion_matrix(cm):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    if len(history["val_loss"]) > 0:
        plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_acc"], label="Train Accuracy")
    if len(history["val_acc"]) > 0:
        plt.plot(history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # -----------------------------
    # 1. PATHS
    # -----------------------------
    data_dir = "C:\\Users\\natep\\Downloads\\MNIST\\MNIST"
    train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

    model_path = "mnist_softmax_model.npz"
    excel_output_path = "mnist_test_predictions.xlsx"

    # -----------------------------
    # 2. HYPERPARAMETERS
    # -----------------------------
    num_classes = 10
    learning_rate = 0.5
    reg_lambda = 1e-4
    epochs = 20
    batch_size = 128
    val_ratio = 0.1

    # -----------------------------
    # 3. LOAD RAW IDX DATA
    # -----------------------------
    print("Loading MNIST IDX files...")
    train_images = load_idx_images(train_images_path)
    y_all = load_idx_labels(train_labels_path)

    test_images = load_idx_images(test_images_path)
    y_test = load_idx_labels(test_labels_path)

    print(f"Raw train images shape: {train_images.shape}")
    print(f"Raw train labels shape: {y_all.shape}")
    print(f"Raw test images shape:  {test_images.shape}")
    print(f"Raw test labels shape:  {y_test.shape}")

    # -----------------------------
    # 4. VISUAL VERIFICATION
    # -----------------------------
    show_samples(train_images, y_all, num_samples=10)

    # -----------------------------
    # 5. PREPROCESS
    # -----------------------------
    X_all = preprocess_images(train_images, normalize=True, flatten=True)
    X_test = preprocess_images(test_images, normalize=True, flatten=True)

    # -----------------------------
    # 6. TRAIN / VAL SPLIT
    # -----------------------------
    X_train, y_train, X_val, y_val = split_train_val(X_all, y_all, val_ratio=val_ratio)

    print(f"Train split: {len(X_train)}")
    print(f"Val split:   {len(X_val)}")
    print(f"Test split:  {len(X_test)}")

    # -----------------------------
    # 7. BUILD MODEL
    # -----------------------------
    model = SoftmaxLogisticRegression(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        lr=learning_rate,
        reg_lambda=reg_lambda
    )

    # -----------------------------
    # 8. TRAIN
    # -----------------------------
    history, train_time = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    print(f"\nTraining time: {train_time:.4f} seconds")

    # -----------------------------
    # 9. SAVE MODEL
    # -----------------------------
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Optional consistency check
    loaded_model = SoftmaxLogisticRegression.load(model_path)
    same_pred = np.all(model.predict(X_test) == loaded_model.predict(X_test))
    print(f"Reload consistency check: {same_pred}")

    # -----------------------------
    # 10. TEST
    # -----------------------------
    test_start = time.perf_counter()
    y_test_pred = loaded_model.predict(X_test)
    test_time = time.perf_counter() - test_start

    test_acc = np.mean(y_test_pred == y_test)
    cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(num_classes))

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Testing time: {test_time:.4f} seconds")
    print("Confusion Matrix:")
    print(cm)

    # -----------------------------
    # 11. EXPORT EXCEL
    # -----------------------------
    export_predictions_to_excel(y_test, y_test_pred, excel_output_path)

    # -----------------------------
    # 12. PLOTS
    # -----------------------------
    plot_training_history(history)
    plot_confusion_matrix(cm)

    # -----------------------------
    # 13. REPORT-READY SUMMARY
    # -----------------------------
    print("\n===== REPORT SUMMARY =====")
    print("Dataset: MNIST IDX format")
    print(f"Input image size: {train_images.shape[1]} x {train_images.shape[2]}")
    print("Preprocessing: normalize to [0,1], flatten 28x28 image to 784-dimensional vector")
    print(f"Training split: {len(X_train)}")
    print(f"Validation split: {len(X_val)}")
    print(f"Testing split: {len(X_test)}")
    print("Model: Multiclass logistic regression (softmax regression)")
    print(f"Parameters: W shape = {model.W.shape}, b shape = {model.b.shape}")
    print(f"Optimizer: Mini-batch gradient descent")
    print(f"Learning rate: {learning_rate}")
    print(f"Regularization: L2, lambda = {reg_lambda}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Training time (s): {train_time:.4f}")
    print(f"Testing time (s): {test_time:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()