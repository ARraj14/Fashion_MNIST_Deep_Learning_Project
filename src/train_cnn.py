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
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
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
print("FASHION-MNIST CONVOLUTIONAL NEURAL NETWORK")
print("=" * 70)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("\nOriginal training shape:", x_train.shape)
print("Original test shape    :", x_test.shape)


# ---------------------------------------------------------
# Normalize images
# ---------------------------------------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# CNN requires a channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("\nCNN training shape:", x_train.shape)
print("CNN test shape    :", x_test.shape)

print(f"Normalized range  : [{x_train.min():.4f}, {x_train.max():.4f}]")


# ---------------------------------------------------------
# Build CNN
# ---------------------------------------------------------
model = Sequential(
    [
        tf.keras.Input(shape=(28, 28, 1)),
        Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),
        BatchNormalization(),
        Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),
        BatchNormalization(),
        Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.30),
        Flatten(),
        Dense(
            128,
            activation="relu",
        ),
        Dropout(0.40),
        Dense(
            10,
            activation="softmax",
        ),
    ],
    name="fashion_mnist_cnn",
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


print("\n" + "=" * 70)
print("CNN ARCHITECTURE")
print("=" * 70)

model.summary()


# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1,
)


# ---------------------------------------------------------
# Train CNN
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING CNN")
print("=" * 70)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.20,
    epochs=25,
    batch_size=128,
    callbacks=[
        early_stopping,
        reduce_lr,
    ],
    verbose=1,
)


# ---------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("CNN TEST SET EVALUATION")
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
prediction_probabilities = model.predict(
    x_test,
    batch_size=128,
    verbose=0,
)

y_pred = np.argmax(
    prediction_probabilities,
    axis=1,
)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print(f"\nScikit-learn Accuracy: {accuracy:.6f}")


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------
report_text = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
)

report_dict = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    output_dict=True,
)

print("\n" + "=" * 70)
print("CNN CLASSIFICATION REPORT")
print("=" * 70)

print(report_text)


with open(
    METRICS_DIR / "cnn_classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report_text)


with open(
    METRICS_DIR / "cnn_classification_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report_dict,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(
    y_test,
    y_pred,
)

np.savetxt(
    METRICS_DIR / "cnn_confusion_matrix.csv",
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

plt.title("CNN Confusion Matrix")

plt.xlabel("Predicted Class")

plt.ylabel("True Class")

plt.xticks(rotation=45, ha="right")

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_cnn_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Accuracy curves
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

plt.title("CNN Training and Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "10_cnn_accuracy_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Loss curves
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

plt.title("CNN Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "11_cnn_loss_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("CNN PER-CLASS ACCURACY")
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
confidence = np.max(
    prediction_probabilities,
    axis=1,
)

correct_mask = y_pred == y_test
incorrect_mask = y_pred != y_test

correct_confidence = confidence[correct_mask]

incorrect_confidence = confidence[incorrect_mask]

mean_correct_confidence = float(correct_confidence.mean())

mean_incorrect_confidence = float(incorrect_confidence.mean())


print("\n" + "=" * 70)
print("CNN CONFIDENCE ANALYSIS")
print("=" * 70)

print(f"Mean confidence — correct predictions   : {mean_correct_confidence:.4f}")

print(f"Mean confidence — incorrect predictions : {mean_incorrect_confidence:.4f}")

print(f"Number of correct predictions   : {correct_mask.sum()}")

print(f"Number of incorrect predictions : {incorrect_mask.sum()}")


# ---------------------------------------------------------
# Misclassification analysis
# ---------------------------------------------------------
incorrect_indices = np.where(y_pred != y_test)[0]

misclassification_pairs = {}

for index in incorrect_indices:
    true_name = CLASS_NAMES[y_test[index]]

    predicted_name = CLASS_NAMES[y_pred[index]]

    pair = f"{true_name} -> {predicted_name}"

    misclassification_pairs[pair] = misclassification_pairs.get(pair, 0) + 1


sorted_pairs = sorted(
    misclassification_pairs.items(),
    key=lambda x: x[1],
    reverse=True,
)


print("\n" + "=" * 70)
print("TOP 10 CNN MISCLASSIFICATIONS")
print("=" * 70)

for pair, count in sorted_pairs[:10]:
    print(f"{pair:<35}: {count}")


# ---------------------------------------------------------
# Incorrect prediction figure
# ---------------------------------------------------------
number_to_plot = min(20, len(incorrect_indices))

selected_errors = incorrect_indices[:number_to_plot]

fig, axes = plt.subplots(
    4,
    5,
    figsize=(13, 11),
)

for ax, index in zip(
    axes.flat,
    selected_errors,
):
    ax.imshow(
        x_test[index].squeeze(),
        cmap="gray",
    )

    true_label = CLASS_NAMES[y_test[index]]

    predicted_label = CLASS_NAMES[y_pred[index]]

    conf = confidence[index] * 100

    ax.set_title(
        f"True: {true_label}\nPred: {predicted_label}\n{conf:.1f}%",
        fontsize=8,
    )

    ax.axis("off")


plt.suptitle(
    "Examples of CNN Misclassifications",
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "12_cnn_misclassified_examples.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Correct prediction examples
# ---------------------------------------------------------
correct_indices = np.where(y_pred == y_test)[0]

selected_correct = correct_indices[:20]

fig, axes = plt.subplots(
    4,
    5,
    figsize=(13, 11),
)

for ax, index in zip(
    axes.flat,
    selected_correct,
):
    ax.imshow(
        x_test[index].squeeze(),
        cmap="gray",
    )

    label = CLASS_NAMES[y_test[index]]

    conf = confidence[index] * 100

    ax.set_title(
        f"{label}\nConfidence: {conf:.1f}%",
        fontsize=8,
    )

    ax.axis("off")


plt.suptitle(
    "Examples of Correct CNN Predictions",
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "13_cnn_correct_predictions.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Confidence distributions
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.hist(
    correct_confidence,
    bins=30,
    alpha=0.6,
    label="Correct",
)

plt.hist(
    incorrect_confidence,
    bins=30,
    alpha=0.6,
    label="Incorrect",
)

plt.title("CNN Prediction Confidence Distribution")

plt.xlabel("Prediction Confidence")

plt.ylabel("Number of Predictions")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "14_cnn_confidence_distribution.png",
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
    METRICS_DIR / "cnn_training_history.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        history_data,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Best epoch
# ---------------------------------------------------------
best_epoch = int(np.argmin(history.history["val_loss"]) + 1)


# ---------------------------------------------------------
# Save CNN summary
# ---------------------------------------------------------
cnn_metrics = {
    "model": "Convolutional Neural Network",
    "seed": SEED,
    "epochs_completed": len(history.history["loss"]),
    "best_epoch": best_epoch,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "correct_predictions": int(correct_mask.sum()),
    "incorrect_predictions": int(incorrect_mask.sum()),
    "mean_correct_confidence": mean_correct_confidence,
    "mean_incorrect_confidence": mean_incorrect_confidence,
    "per_class_accuracy": per_class_accuracy,
    "top_misclassification_pairs": [
        {
            "pair": pair,
            "count": int(count),
        }
        for pair, count in sorted_pairs[:10]
    ],
}


with open(
    METRICS_DIR / "cnn_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        cnn_metrics,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
model_path = MODEL_DIR / "fashion_mnist_cnn.keras"

model.save(model_path)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("CNN FILES GENERATED")
print("=" * 70)

print("\nModel:")
print("fashion_mnist_cnn.keras")

print("\nFigures:")
print("09_cnn_confusion_matrix.png")
print("10_cnn_accuracy_curve.png")
print("11_cnn_loss_curve.png")
print("12_cnn_misclassified_examples.png")
print("13_cnn_correct_predictions.png")
print("14_cnn_confidence_distribution.png")

print("\nMetrics:")
print("cnn_metrics.json")
print("cnn_training_history.json")
print("cnn_classification_report.json")
print("cnn_classification_report.txt")
print("cnn_confusion_matrix.csv")

print("\nCNN training and evaluation completed successfully.")
