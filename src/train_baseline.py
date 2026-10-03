import os
import logging

# Suppress normal TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Temporarily suppress STDERR only while TensorFlow initializes
_saved_stderr = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)

os.dup2(_devnull, 2)

try:
    import tensorflow as tf
finally:
    os.dup2(_saved_stderr, 2)
    os.close(_saved_stderr)
    os.close(_devnull)

# Suppress TensorFlow Python warnings such as the native-Windows GPU notice
tf.get_logger().setLevel(logging.ERROR)

from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


print("=" * 70)
print("FASHION-MNIST BASELINE NEURAL NETWORK")
print("=" * 70)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("\nOriginal training shape:", x_train.shape)
print("Original test shape    :", x_test.shape)


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------
print("\nNormalizing pixel values from [0, 255] to [0, 1]...")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"Training minimum after normalization: {x_train.min():.4f}")
print(f"Training maximum after normalization: {x_train.max():.4f}")


# ---------------------------------------------------------
# Build baseline neural network
# ---------------------------------------------------------
model = Sequential(
    [
        tf.keras.Input(shape=(28, 28)),
        Flatten(),
        Dense(
            128,
            activation="relu",
        ),
        Dropout(0.30),
        Dense(
            64,
            activation="relu",
        ),
        Dropout(0.20),
        Dense(
            10,
            activation="softmax",
        ),
    ],
    name="fashion_mnist_baseline",
)


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

model.summary()


# ---------------------------------------------------------
# Early stopping
# ---------------------------------------------------------
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING BASELINE MODEL")
print("=" * 70)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.20,
    epochs=20,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1,
)


# ---------------------------------------------------------
# Evaluate test set
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0,
)

print(f"Test Loss     : {test_loss:.6f}")
print(f"Test Accuracy : {test_accuracy:.6f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------
prediction_probabilities = model.predict(x_test, verbose=0)

y_pred = np.argmax(
    prediction_probabilities,
    axis=1,
)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nScikit-learn Accuracy: {accuracy:.6f}")


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------
report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    output_dict=True,
)

report_text = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report_text)


with open(
    METRICS_DIR / "baseline_classification_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report,
        file,
        indent=4,
    )


with open(
    METRICS_DIR / "baseline_classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report_text)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

np.savetxt(
    METRICS_DIR / "baseline_confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
)


plt.figure(figsize=(11, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)

plt.title("Baseline Neural Network Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("True Class")

plt.xticks(rotation=45, ha="right")

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_baseline_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Accuracy curve
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
)

plt.title("Baseline Model Training and Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_baseline_accuracy_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Loss curve
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
)

plt.title("Baseline Model Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "08_baseline_loss_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Save training history
# ---------------------------------------------------------
history_data = {}

for key, values in history.history.items():
    history_data[key] = [float(value) for value in values]

with open(
    METRICS_DIR / "baseline_training_history.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        history_data,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Calculate per-class accuracy
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

per_class_accuracy = {}

for class_id, class_name in enumerate(CLASS_NAMES):
    mask = y_test == class_id

    class_accuracy = (y_pred[mask] == y_test[mask]).mean()

    per_class_accuracy[class_name] = float(class_accuracy)

    print(f"{class_name:<15}: {class_accuracy * 100:.2f}%")


# ---------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------
prediction_confidence = np.max(
    prediction_probabilities,
    axis=1,
)

correct_mask = y_pred == y_test

incorrect_mask = y_pred != y_test

mean_correct_confidence = prediction_confidence[correct_mask].mean()

mean_incorrect_confidence = prediction_confidence[incorrect_mask].mean()


print("\n" + "=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)

print(f"Mean confidence — correct predictions   : {mean_correct_confidence:.4f}")

print(f"Mean confidence — incorrect predictions : {mean_incorrect_confidence:.4f}")

print(f"Number of correct predictions   : {correct_mask.sum()}")

print(f"Number of incorrect predictions : {incorrect_mask.sum()}")


# ---------------------------------------------------------
# Save summary metrics
# ---------------------------------------------------------
best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

summary_metrics = {
    "model": "Baseline Dense Neural Network",
    "seed": SEED,
    "epochs_completed": len(history.history["loss"]),
    "best_epoch": best_epoch,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "correct_predictions": int(correct_mask.sum()),
    "incorrect_predictions": int(incorrect_mask.sum()),
    "mean_correct_confidence": float(mean_correct_confidence),
    "mean_incorrect_confidence": float(mean_incorrect_confidence),
    "per_class_accuracy": per_class_accuracy,
}


with open(
    METRICS_DIR / "baseline_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary_metrics,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
model_path = MODEL_DIR / "fashion_mnist_baseline.keras"

model.save(model_path)


print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print("\nModel:")
print("fashion_mnist_baseline.keras")

print("\nFigures:")
print("06_baseline_confusion_matrix.png")
print("07_baseline_accuracy_curve.png")
print("08_baseline_loss_curve.png")

print("\nMetrics:")
print("baseline_metrics.json")
print("baseline_training_history.json")
print("baseline_classification_report.json")
print("baseline_classification_report.txt")
print("baseline_confusion_matrix.csv")

print("\nBaseline neural network training completed successfully.")
