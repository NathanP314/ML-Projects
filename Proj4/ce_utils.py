import os
import json
import time
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# FEATURE EXTRACTION (pure numpy — no skimage needed)
# ============================================================

def _compute_gradients(img):
    """Compute gradient magnitude and orientation using Sobel-like filters."""
    # Sobel kernels via numpy convolution (valid region)
    gy = np.zeros_like(img)
    gx = np.zeros_like(img)
    # Central differences (fast, no scipy needed)
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)  # radians in [-pi, pi]
    # Map to [0, pi) for unsigned gradients
    orientation = orientation % np.pi
    return magnitude, orientation


def _hog_features(img, orientations=9, pixels_per_cell=8):
    """
    Simplified HOG: compute oriented gradient histograms per cell.
    No block normalization (standardization later handles scale).
    """
    mag, ori = _compute_gradients(img)
    h, w = img.shape
    cells_y = h // pixels_per_cell
    cells_x = w // pixels_per_cell
    hog_hist = []

    bin_width = np.pi / orientations

    for cy in range(cells_y):
        for cx in range(cells_x):
            r0 = cy * pixels_per_cell
            c0 = cx * pixels_per_cell
            cell_mag = mag[r0:r0+pixels_per_cell, c0:c0+pixels_per_cell]
            cell_ori = ori[r0:r0+pixels_per_cell, c0:c0+pixels_per_cell]

            hist = np.zeros(orientations, dtype=np.float32)
            bin_idx = np.clip((cell_ori / bin_width).astype(int), 0, orientations - 1)
            for b in range(orientations):
                hist[b] = np.sum(cell_mag[bin_idx == b])

            # L2 normalize per cell
            norm = np.sqrt(np.sum(hist**2) + 1e-6)
            hist /= norm
            hog_hist.append(hist)

    return np.concatenate(hog_hist)


def _sobel_edges(img):
    """Compute edge magnitude using central differences."""
    gy = np.zeros_like(img)
    gx = np.zeros_like(img)
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    return np.sqrt(gx**2 + gy**2)


def extract_features(img_array):
    """
    Extract a rich feature vector from a 2D grayscale image array (values in [0,1]).
    Uses only numpy — no skimage or scipy required.
    """
    features = []

    # --- 1) HOG features (edge/gradient structure) ---
    hog_feat = _hog_features(img_array, orientations=9, pixels_per_cell=8)
    features.append(hog_feat)

    # --- 2) Intensity histogram (brightness distribution) ---
    hist, _ = np.histogram(img_array, bins=32, range=(0.0, 1.0))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-12)
    features.append(hist)

    # --- 3) Spatial block statistics (spatial layout) ---
    h, w = img_array.shape
    bh, bw = h // 4, w // 4
    block_means = []
    block_stds = []
    for i in range(4):
        for j in range(4):
            block = img_array[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            block_means.append(np.mean(block))
            block_stds.append(np.std(block))
    features.append(np.array(block_means, dtype=np.float32))
    features.append(np.array(block_stds, dtype=np.float32))

    # --- 4) Edge statistics ---
    edges = _sobel_edges(img_array)
    features.append(np.array([
        np.mean(edges),
        np.std(edges),
        np.max(edges),
        np.percentile(edges, 25),
        np.percentile(edges, 50),
        np.percentile(edges, 75),
    ], dtype=np.float32))

    # --- 5) Global statistics ---
    features.append(np.array([
        np.mean(img_array),
        np.std(img_array),
        np.min(img_array),
        np.max(img_array),
        np.percentile(img_array, 10),
        np.percentile(img_array, 90),
    ], dtype=np.float32))

    return np.concatenate(features).astype(np.float32)


def load_single_image(image_path, expected_shape=None):
    img = Image.open(image_path).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32)

    if expected_shape is not None and arr.shape != tuple(expected_shape):
        raise ValueError(f"Image {image_path} has shape {arr.shape}, expected {expected_shape}")

    arr = arr / 255.0
    feat = extract_features(arr)
    return feat, arr.shape


def load_celegans_dataset(base_dir):
    """
    Expects:
    base_dir/
        0/*.png
        1/*.png
    """
    X = []
    y = []
    file_paths = []

    class_counts = {}

    for label_str in ["0", "1"]:
        class_dir = os.path.join(base_dir, label_str)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(".png")])
        class_counts[int(label_str)] = len(files)

        for fname in files:
            full_path = os.path.join(class_dir, fname)
            feat, shape = load_single_image(full_path)
            X.append(feat)
            y.append(int(label_str))
            file_paths.append(full_path)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, file_paths, shape, class_counts


# ============================================================
# FEATURE STANDARDIZATION
# ============================================================

def standardize_fit(X):
    """Compute mean and std from training data."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8  # avoid division by zero
    return mean, std


def standardize_transform(X, mean, std):
    """Apply standardization."""
    return (X - mean) / std


# ============================================================
# DATA SPLITTING
# ============================================================

def stratified_split_indices(y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []
    test_idx = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob):
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def compute_metrics(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def confusion_matrix_dataframe(metrics):
    return pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]]
        ],
        index=["Actual 0 (No Worm)", "Actual 1 (Worm)"],
        columns=["Predicted 0 (No Worm)", "Predicted 1 (Worm)"]
    )


def train_logistic_regression(
    X_train, y_train,
    X_val, y_val,
    learning_rate=0.01,
    epochs=100,
    batch_size=64,
    seed=42,
    l2_lambda=0.01,
    lr_decay=0.995
):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape

    # Xavier initialization instead of zeros
    w = rng.standard_normal(n_features).astype(np.float32) * np.sqrt(2.0 / n_features)
    b = 0.0

    history = []
    best_val_acc = 0.0
    best_w = w.copy()
    best_b = b
    patience = 50
    no_improve = 0

    current_lr = learning_rate

    for epoch in range(epochs):
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        X_train_shuf = X_train[indices]
        y_train_shuf = y_train[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            xb = X_train_shuf[start:end]
            yb = y_train_shuf[start:end]

            logits = np.dot(xb, w) + b
            probs = sigmoid(logits)

            error = probs - yb
            dw = np.dot(xb.T, error) / len(xb) + l2_lambda * w  # L2 regularization
            db = np.mean(error)

            w -= current_lr * dw
            b -= current_lr * db

        # Learning rate decay
        current_lr *= lr_decay

        train_probs = sigmoid(np.dot(X_train, w) + b)
        val_probs = sigmoid(np.dot(X_val, w) + b)

        train_loss = binary_cross_entropy(y_train, train_probs)
        val_loss = binary_cross_entropy(y_val, val_probs)

        train_pred = (train_probs >= 0.5).astype(np.int32)
        val_pred = (val_probs >= 0.5).astype(np.int32)

        train_acc = np.mean(train_pred == y_train)
        val_acc = np.mean(val_pred == y_val)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "lr": float(current_lr)
        })

        # Early stopping with best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_w = w.copy()
            best_b = b
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
                f"lr={current_lr:.6f} | best_val={best_val_acc:.4f}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
            break

    # Return best model weights
    return best_w, best_b, history


def predict_probs(X, w, b):
    return sigmoid(np.dot(X, w) + b)


def predict_labels(X, w, b, threshold=0.5):
    probs = predict_probs(X, w, b)
    return (probs >= threshold).astype(np.int32)


def save_model(model_path, w, b, image_shape, mean=None, std=None):
    save_dict = dict(w=w, b=b, image_shape=np.array(image_shape))
    if mean is not None:
        save_dict["feat_mean"] = mean
        save_dict["feat_std"] = std
    np.savez(model_path, **save_dict)


def load_model(model_path):
    data = np.load(model_path)
    w = data["w"]
    b = float(data["b"])
    image_shape = tuple(data["image_shape"].tolist())
    mean = data.get("feat_mean", None)
    std = data.get("feat_std", None)
    return w, b, image_shape, mean, std


def create_visual_verification(base_dir, output_path, samples_per_class=4, seed=42):
    rng = random.Random(seed)

    fig, axes = plt.subplots(2, samples_per_class, figsize=(12, 6))

    for row, label_str in enumerate(["0", "1"]):
        class_dir = os.path.join(base_dir, label_str)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]
        chosen = rng.sample(files, min(samples_per_class, len(files)))

        for col in range(samples_per_class):
            ax = axes[row, col]
            if col < len(chosen):
                img_path = os.path.join(class_dir, chosen[col])
                img = Image.open(img_path).convert("L")
                ax.imshow(img, cmap="gray")
                ax.set_title(f"Class {label_str}\n{chosen[col]}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_report_table_csv(
    output_path,
    total_count,
    class_counts,
    train_count,
    val_count,
    test_count,
    image_shape,
    learning_rate,
    epochs,
    batch_size,
    training_time,
    testing_time
):
    rows = [
        ["Visual verification of input data", "Saved sample image grid to outputs/celegans_visual_check.png"],
        ["Class labels", "0 = no worm, 1 = worm"],
        ["Total dataset size", total_count],
        ["Class 0 count", class_counts.get(0, 0)],
        ["Class 1 count", class_counts.get(1, 0)],
        ["Training split", train_count],
        ["Validation split", val_count],
        ["Testing split", test_count],
        ["Input image size", f"{image_shape[0]} x {image_shape[1]}"],
        ["Feature extraction", "HOG (9 orientations, 8x8 cells, 2x2 blocks) + 32-bin intensity histogram + 4x4 spatial block stats + edge stats + global stats"],
        ["Feature standardization", "Zero-mean, unit-variance (fit on train set)"],
        ["Logistic regression regularization", "L2 (weight decay)"],
        ["Optimizer type", "Mini-batch gradient descent with LR decay and early stopping"],
        ["Optimizer parameters", f"learning_rate={learning_rate}, epochs={epochs}, batch_size={batch_size}"],
        ["Training execution time (seconds)", training_time],
        ["Testing execution time (seconds)", testing_time]
    ]

    df = pd.DataFrame(rows, columns=["Category", "Value"])
    df.to_csv(output_path, index=False)


def write_inference_excel(output_path, filenames, labels):
    df = pd.DataFrame({
        "filename": filenames,
        "predicted_label": labels
    })

    counts = pd.DataFrame({
        "label": [0, 1],
        "count": [
            int(np.sum(np.array(labels) == 0)),
            int(np.sum(np.array(labels) == 1))
        ]
    })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
        counts.to_excel(writer, index=False, sheet_name="label_counts")