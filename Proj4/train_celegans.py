import os
import json
import time
import pandas as pd

from ce_utils import (
    load_celegans_dataset,
    stratified_split_indices,
    standardize_fit,
    standardize_transform,
    train_logistic_regression,
    predict_labels,
    compute_metrics,
    confusion_matrix_dataframe,
    save_model,
    create_visual_verification,
    write_report_table_csv
)


# =========================
# SETTINGS
# =========================


DATASET_DIR = "C:\\Users\\natep\\Downloads\\Celegans_ModelGen"
MODEL_PATH = "models/celegans_model.npz"
VISUAL_CHECK_PATH = "outputs/celegans_visual_check.png"
REPORT_TABLE_PATH = "outputs/celegans_report_table.csv"
CONFUSION_MATRIX_PATH = "outputs/celegans_confusion_matrix.csv"
METRICS_JSON_PATH = "outputs/celegans_metrics.json"
TRAIN_HISTORY_PATH = "outputs/celegans_training_history.csv"

LEARNING_RATE = 0.05
EPOCHS = 500
BATCH_SIZE = 32
SEED = 42
L2_LAMBDA = 0.001
LR_DECAY = 0.998


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Loading dataset (with feature extraction)...")
    X, y, file_paths, image_shape, class_counts = load_celegans_dataset(DATASET_DIR)

    total_count = len(y)
    print(f"Total images: {total_count}")
    print(f"Image shape: {image_shape}")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Class counts: {class_counts}")

    print("Creating visual verification image...")
    create_visual_verification(DATASET_DIR, VISUAL_CHECK_PATH)

    print("Creating stratified train/val/test split...")
    train_idx, val_idx, test_idx = stratified_split_indices(
        y,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=SEED
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize features (fit on train, transform all)
    print("Standardizing features...")
    feat_mean, feat_std = standardize_fit(X_train)
    X_train = standardize_transform(X_train, feat_mean, feat_std)
    X_val = standardize_transform(X_val, feat_mean, feat_std)
    X_test = standardize_transform(X_test, feat_mean, feat_std)

    print(f"Train size: {len(y_train)}")
    print(f"Val size:   {len(y_val)}")
    print(f"Test size:  {len(y_test)}")

    print("Training logistic regression...")
    train_start = time.perf_counter()

    w, b, history = train_logistic_regression(
        X_train, y_train,
        X_val, y_val,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        l2_lambda=L2_LAMBDA,
        lr_decay=LR_DECAY
    )

    train_end = time.perf_counter()
    training_time = train_end - train_start

    print(f"Training time: {training_time:.4f} seconds")

    print("Testing model...")
    test_start = time.perf_counter()

    y_pred = predict_labels(X_test, w, b, threshold=0.5)
    metrics = compute_metrics(y_test, y_pred)

    test_end = time.perf_counter()
    testing_time = test_end - test_start

    print(f"Testing time: {testing_time:.4f} seconds")

    print("Saving model...")
    save_model(MODEL_PATH, w, b, image_shape, mean=feat_mean, std=feat_std)

    print("Saving training history...")
    pd.DataFrame(history).to_csv(TRAIN_HISTORY_PATH, index=False)

    print("Saving confusion matrix...")
    cm_df = confusion_matrix_dataframe(metrics)
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    print("Saving metrics json...")
    metrics_out = dict(metrics)
    metrics_out["training_time_seconds"] = training_time
    metrics_out["testing_time_seconds"] = testing_time
    metrics_out["image_shape"] = list(image_shape)
    metrics_out["class_counts"] = class_counts
    metrics_out["feature_vector_size"] = int(X.shape[1])

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_out, f, indent=4)

    print("Saving report table csv...")
    write_report_table_csv(
        REPORT_TABLE_PATH,
        total_count=total_count,
        class_counts=class_counts,
        train_count=len(y_train),
        val_count=len(y_val),
        test_count=len(y_test),
        image_shape=image_shape,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        training_time=training_time,
        testing_time=testing_time
    )

    print("\nDone.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Visual check saved to: {VISUAL_CHECK_PATH}")
    print(f"Report table saved to: {REPORT_TABLE_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Metrics saved to: {METRICS_JSON_PATH}")
    print(f"Training history saved to: {TRAIN_HISTORY_PATH}")

    print("\nTest Metrics:")
    print(f"TN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"TP: {metrics['tp']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()