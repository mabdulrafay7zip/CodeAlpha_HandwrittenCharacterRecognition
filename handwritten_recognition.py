"""
Handwritten Character (Digit) Recognition — CodeAlpha Machine Learning Internship
Author: Muhammad Abdul Rafay — CodeAlpha ML Intern

Recognises handwritten digits (0-9). Dataset: scikit-learn's Digits dataset
(1,797 real handwritten digit images, 8x8 pixels, originally from the UCI /
MNIST family of handwritten-digit data) — used because it ships with
scikit-learn, so the project runs anywhere with no download required.

An MLP (multi-layer perceptron neural network) is the main model; Logistic
Regression and SVM are trained as baselines for comparison. This is the
scikit-learn equivalent of the classic MNIST neural-network task: to scale
it up to full 28x28 MNIST, swap load_digits() for fetch_openml('mnist_784')
— the rest of the pipeline is identical.

Run:
    pip install -r requirements.txt
    python handwritten_recognition.py
Outputs:
    outputs/metrics.json
    outputs/sample_predictions.png   - test digits with predicted labels
    outputs/confusion_matrix_mlp.png
    outputs/model_comparison.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def save_sample_predictions(images, y_true, y_pred, n: int = 20) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for ax, img, true_label, pred_label in zip(axes.ravel(), images[:n], y_true[:n], y_pred[:n]):
        ax.imshow(img, cmap="gray_r", interpolation="nearest")
        colour = "green" if pred_label == true_label else "red"
        ax.set_title(f"Pred: {pred_label} (true: {true_label})", color=colour, fontsize=10)
        ax.axis("off")
    fig.suptitle("Handwritten Digit Recognition — Sample Test Predictions (MLP)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=150)
    plt.close()


def save_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
    plt.title("Confusion Matrix — MLP Neural Network")
    plt.xlabel("Predicted digit")
    plt.ylabel("Actual digit")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_mlp.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X, y, images = digits.data, digits.target, digits.images
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features (8x8 images), 10 classes (0-9)")

    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, images, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="rbf"),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
    }

    results = []
    mlp_pred = None
    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = round(accuracy_score(y_test, pred), 4)
        results.append({"model": name, "test_accuracy": acc})
        print(f"--- {name}: test accuracy = {acc:.4f} ---")
        if name == "MLP Neural Network":
            mlp_pred = pred
            print(classification_report(y_test, pred, digits=4))

    save_sample_predictions(img_test, y_test, mlp_pred)
    save_confusion_matrix(y_test, mlp_pred)

    pd.DataFrame(results).set_index("model").plot(kind="barh", figsize=(8, 4), xlim=(0, 1), legend=False)
    plt.title("Handwritten Digit Recognition — Test Accuracy by Model")
    plt.xlabel("Test accuracy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== Summary ===")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nPlots and metrics saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
