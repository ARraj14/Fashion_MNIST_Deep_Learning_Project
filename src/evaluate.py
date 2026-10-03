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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.stats import binomtest
from tensorflow.keras.datasets import fashion_mnist


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

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


print("=" * 75)
print("BASELINE VS CNN — DETAILED MODEL COMPARISON")
print("=" * 75)


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
baseline_path = MODEL_DIR / "fashion_mnist_baseline.keras"
cnn_path = MODEL_DIR / "fashion_mnist_cnn.keras"

print("\nLoading trained models...")

baseline_model = tf.keras.models.load_model(baseline_path)
cnn_model = tf.keras.models.load_model(cnn_path)

print("Baseline model loaded.")
print("CNN model loaded.")


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------
(_, _), (x_test, y_test) = fashion_mnist.load_data()

baseline_x = x_test.astype("float32") / 255.0

cnn_x = baseline_x[..., np.newaxis]

print(f"\nTest samples: {len(y_test)}")


# ---------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------
print("\nGenerating baseline predictions...")

baseline_prob = baseline_model.predict(
    baseline_x,
    batch_size=128,
    verbose=0,
)

baseline_pred = np.argmax(
    baseline_prob,
    axis=1,
)


print("Generating CNN predictions...")

cnn_prob = cnn_model.predict(
    cnn_x,
    batch_size=128,
    verbose=0,
)

cnn_pred = np.argmax(
    cnn_prob,
    axis=1,
)


# ---------------------------------------------------------
# Basic comparison
# ---------------------------------------------------------
baseline_correct = baseline_pred == y_test
cnn_correct = cnn_pred == y_test

baseline_accuracy = baseline_correct.mean()
cnn_accuracy = cnn_correct.mean()

baseline_errors = (~baseline_correct).sum()
cnn_errors = (~cnn_correct).sum()

accuracy_gain = cnn_accuracy - baseline_accuracy

error_reduction = (baseline_errors - cnn_errors) / baseline_errors


print("\n" + "=" * 75)
print("OVERALL PERFORMANCE COMPARISON")
print("=" * 75)

print(f"Baseline accuracy          : {baseline_accuracy * 100:.2f}%")

print(f"CNN accuracy               : {cnn_accuracy * 100:.2f}%")

print(f"Absolute accuracy gain     : {accuracy_gain * 100:.2f} percentage points")

print(f"Baseline errors            : {baseline_errors}")

print(f"CNN errors                 : {cnn_errors}")

print(f"Errors eliminated          : {baseline_errors - cnn_errors}")

print(f"Relative error reduction   : {error_reduction * 100:.2f}%")


# ---------------------------------------------------------
# Paired prediction analysis
# ---------------------------------------------------------
both_correct = (baseline_correct & cnn_correct).sum()

both_wrong = ((~baseline_correct) & (~cnn_correct)).sum()

cnn_fixed_baseline = ((~baseline_correct) & cnn_correct).sum()

baseline_better = (baseline_correct & (~cnn_correct)).sum()


print("\n" + "=" * 75)
print("PAIRED PREDICTION ANALYSIS")
print("=" * 75)

print(f"Both models correct       : {both_correct}")

print(f"Both models wrong         : {both_wrong}")

print(f"CNN fixed baseline error  : {cnn_fixed_baseline}")

print(f"CNN introduced new error  : {baseline_better}")


# ---------------------------------------------------------
# McNemar exact significance test
# ---------------------------------------------------------
discordant = cnn_fixed_baseline + baseline_better

if discordant > 0:
    mcnemar_result = binomtest(
        k=min(cnn_fixed_baseline, baseline_better),
        n=discordant,
        p=0.5,
        alternative="two-sided",
    )

    p_value = float(mcnemar_result.pvalue)

else:
    p_value = 1.0


print(f"\nMcNemar exact-test p-value: {p_value:.10f}")

if p_value < 0.05:
    print("Difference is statistically significant at alpha = 0.05.")
else:
    print("Difference is not statistically significant at alpha = 0.05.")


# ---------------------------------------------------------
# Per-class comparison
# ---------------------------------------------------------
comparison_rows = []

for class_id, class_name in enumerate(CLASS_NAMES):
    mask = y_test == class_id

    baseline_class_acc = (baseline_pred[mask] == y_test[mask]).mean()

    cnn_class_acc = (cnn_pred[mask] == y_test[mask]).mean()

    difference = cnn_class_acc - baseline_class_acc

    comparison_rows.append(
        {
            "Class_ID": class_id,
            "Class_Name": class_name,
            "Baseline_Accuracy": baseline_class_acc,
            "CNN_Accuracy": cnn_class_acc,
            "Improvement": difference,
        }
    )


comparison_df = pd.DataFrame(comparison_rows)


print("\n" + "=" * 75)
print("PER-CLASS MODEL COMPARISON")
print("=" * 75)

display_df = comparison_df.copy()

display_df["Baseline_Accuracy"] *= 100

display_df["CNN_Accuracy"] *= 100

display_df["Improvement"] *= 100

print(
    display_df.to_string(
        index=False,
        formatters={
            "Baseline_Accuracy": "{:.2f}%".format,
            "CNN_Accuracy": "{:.2f}%".format,
            "Improvement": "{:+.2f}".format,
        },
    )
)


comparison_df.to_csv(
    METRICS_DIR / "model_per_class_comparison.csv",
    index=False,
)


# ---------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------
baseline_confidence = np.max(
    baseline_prob,
    axis=1,
)

cnn_confidence = np.max(
    cnn_prob,
    axis=1,
)


confidence_summary = {
    "baseline_mean_confidence": float(baseline_confidence.mean()),
    "cnn_mean_confidence": float(cnn_confidence.mean()),
    "baseline_correct_confidence": float(baseline_confidence[baseline_correct].mean()),
    "baseline_wrong_confidence": float(baseline_confidence[~baseline_correct].mean()),
    "cnn_correct_confidence": float(cnn_confidence[cnn_correct].mean()),
    "cnn_wrong_confidence": float(cnn_confidence[~cnn_correct].mean()),
}


print("\n" + "=" * 75)
print("CONFIDENCE COMPARISON")
print("=" * 75)

for key, value in confidence_summary.items():
    print(f"{key:<35}: {value:.4f}")


# ---------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------
def expected_calibration_error(
    probabilities,
    predictions,
    labels,
    bins=10,
):
    confidences = np.max(
        probabilities,
        axis=1,
    )

    correctness = (predictions == labels).astype(float)

    boundaries = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0
    calibration_rows = []

    for i in range(bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]

        if i == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)

        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = mask.sum()

        if count == 0:
            continue

        avg_confidence = confidences[mask].mean()

        avg_accuracy = correctness[mask].mean()

        gap = abs(avg_confidence - avg_accuracy)

        ece += (count / len(labels)) * gap

        calibration_rows.append(
            {
                "Lower": lower,
                "Upper": upper,
                "Count": int(count),
                "Average_Confidence": float(avg_confidence),
                "Observed_Accuracy": float(avg_accuracy),
            }
        )

    return (
        float(ece),
        pd.DataFrame(calibration_rows),
    )


baseline_ece, baseline_calibration = expected_calibration_error(
    baseline_prob,
    baseline_pred,
    y_test,
)

cnn_ece, cnn_calibration = expected_calibration_error(
    cnn_prob,
    cnn_pred,
    y_test,
)


print("\n" + "=" * 75)
print("CALIBRATION ANALYSIS")
print("=" * 75)

print(f"Baseline ECE : {baseline_ece:.4f}")

print(f"CNN ECE      : {cnn_ece:.4f}")


baseline_calibration.to_csv(
    METRICS_DIR / "baseline_calibration.csv",
    index=False,
)

cnn_calibration.to_csv(
    METRICS_DIR / "cnn_calibration.csv",
    index=False,
)


# ---------------------------------------------------------
# Figure 15 — Overall accuracy
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

model_names = [
    "Baseline NN",
    "CNN",
]

accuracies = [
    baseline_accuracy * 100,
    cnn_accuracy * 100,
]

bars = plt.bar(
    model_names,
    accuracies,
)

plt.ylabel("Test Accuracy (%)")

plt.title("Baseline Neural Network vs CNN")

plt.ylim(
    80,
    100,
)

for bar, value in zip(
    bars,
    accuracies,
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.3,
        f"{value:.2f}%",
        ha="center",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "15_model_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Figure 16 — Per-class comparison
# ---------------------------------------------------------
plot_df = display_df.copy()

x = np.arange(len(CLASS_NAMES))

width = 0.38

plt.figure(figsize=(14, 7))

plt.bar(
    x - width / 2,
    plot_df["Baseline_Accuracy"],
    width,
    label="Baseline",
)

plt.bar(
    x + width / 2,
    plot_df["CNN_Accuracy"],
    width,
    label="CNN",
)

plt.xticks(
    x,
    CLASS_NAMES,
    rotation=40,
    ha="right",
)

plt.ylabel("Accuracy (%)")

plt.title("Per-Class Accuracy: Baseline vs CNN")

plt.ylim(
    60,
    102,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "16_per_class_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Figure 17 — Improvement by class
# ---------------------------------------------------------
improvements = comparison_df["Improvement"] * 100

plt.figure(figsize=(12, 6))

bars = plt.bar(
    CLASS_NAMES,
    improvements,
)

plt.axhline(
    0,
    linewidth=1,
)

plt.ylabel("CNN Improvement (percentage points)")

plt.title("CNN Accuracy Change by Class")

plt.xticks(
    rotation=40,
    ha="right",
)

for bar, value in zip(
    bars,
    improvements,
):
    vertical_alignment = "bottom" if value >= 0 else "top"

    offset = 0.25 if value >= 0 else -0.25

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:+.1f}",
        ha="center",
        va=vertical_alignment,
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "17_per_class_improvement.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Figure 18 — Error count
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

error_values = [
    baseline_errors,
    cnn_errors,
]

bars = plt.bar(
    model_names,
    error_values,
)

plt.title("Number of Incorrect Test Predictions")

plt.ylabel("Incorrect Predictions")

for bar, value in zip(
    bars,
    error_values,
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 20,
        str(value),
        ha="center",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "18_model_error_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Figure 19 — Calibration comparison
# ---------------------------------------------------------
plt.figure(figsize=(8, 7))

plt.plot(
    baseline_calibration["Average_Confidence"],
    baseline_calibration["Observed_Accuracy"],
    marker="o",
    label=f"Baseline (ECE={baseline_ece:.3f})",
)

plt.plot(
    cnn_calibration["Average_Confidence"],
    cnn_calibration["Observed_Accuracy"],
    marker="o",
    label=f"CNN (ECE={cnn_ece:.3f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration",
)

plt.xlabel("Mean Prediction Confidence")

plt.ylabel("Observed Accuracy")

plt.title("Model Calibration Comparison")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "19_model_calibration_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Figure 20 — Cases CNN corrected
# ---------------------------------------------------------
corrected_indices = np.where((~baseline_correct) & cnn_correct)[0]

number_to_show = min(20, len(corrected_indices))

selected = corrected_indices[:number_to_show]

if number_to_show > 0:
    fig, axes = plt.subplots(
        4,
        5,
        figsize=(13, 11),
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, index in zip(
        axes.flat,
        selected,
    ):
        ax.imshow(
            x_test[index],
            cmap="gray",
        )

        true_name = CLASS_NAMES[y_test[index]]

        baseline_name = CLASS_NAMES[baseline_pred[index]]

        cnn_name = CLASS_NAMES[cnn_pred[index]]

        ax.set_title(
            f"True: {true_name}\nBase: {baseline_name}\nCNN: {cnn_name}",
            fontsize=8,
        )

        ax.axis("off")

    plt.suptitle(
        "Examples Where CNN Corrected Baseline Errors",
        fontsize=14,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "20_cnn_corrected_baseline_errors.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------
# Save overall comparison
# ---------------------------------------------------------
comparison_summary = {
    "baseline_accuracy": float(baseline_accuracy),
    "cnn_accuracy": float(cnn_accuracy),
    "absolute_accuracy_gain": float(accuracy_gain),
    "baseline_errors": int(baseline_errors),
    "cnn_errors": int(cnn_errors),
    "errors_eliminated": int(baseline_errors - cnn_errors),
    "relative_error_reduction": float(error_reduction),
    "both_correct": int(both_correct),
    "both_wrong": int(both_wrong),
    "cnn_fixed_baseline_errors": int(cnn_fixed_baseline),
    "cnn_new_errors": int(baseline_better),
    "mcnemar_exact_p_value": p_value,
    "baseline_ece": baseline_ece,
    "cnn_ece": cnn_ece,
    "confidence_summary": confidence_summary,
}


with open(
    METRICS_DIR / "model_comparison_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        comparison_summary,
        file,
        indent=4,
    )


print("\n" + "=" * 75)
print("COMPARISON FILES GENERATED")
print("=" * 75)

print("\nFigures:")
print("15_model_accuracy_comparison.png")
print("16_per_class_accuracy_comparison.png")
print("17_per_class_improvement.png")
print("18_model_error_comparison.png")
print("19_model_calibration_comparison.png")
print("20_cnn_corrected_baseline_errors.png")

print("\nMetrics:")
print("model_per_class_comparison.csv")
print("model_comparison_summary.json")
print("baseline_calibration.csv")
print("cnn_calibration.csv")

print("\nDetailed model comparison completed successfully.")
