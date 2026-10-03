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
import tensorflow as tf
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
print("FASHION-MNIST CNN — DEEP ERROR ANALYSIS")
print("=" * 75)


# ---------------------------------------------------------
# Load CNN and test data
# ---------------------------------------------------------
cnn_model = tf.keras.models.load_model(MODEL_DIR / "fashion_mnist_cnn.keras")

(_, _), (x_test, y_test) = fashion_mnist.load_data()

x_normalized = x_test.astype("float32") / 255.0

x_cnn = x_normalized[..., np.newaxis]


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------
probabilities = cnn_model.predict(
    x_cnn,
    batch_size=128,
    verbose=0,
)

predictions = np.argmax(
    probabilities,
    axis=1,
)

confidence = np.max(
    probabilities,
    axis=1,
)

correct = predictions == y_test
incorrect = ~correct


print(f"\nCorrect predictions   : {correct.sum()}")

print(f"Incorrect predictions : {incorrect.sum()}")


# ---------------------------------------------------------
# All misclassification pairs
# ---------------------------------------------------------
pair_rows = []

for true_id in range(10):
    for predicted_id in range(10):
        if true_id == predicted_id:
            continue

        count = np.sum((y_test == true_id) & (predictions == predicted_id))

        if count > 0:
            pair_rows.append(
                {
                    "True_Class": CLASS_NAMES[true_id],
                    "Predicted_Class": CLASS_NAMES[predicted_id],
                    "Count": int(count),
                }
            )


pair_df = pd.DataFrame(pair_rows)

pair_df = pair_df.sort_values(
    "Count",
    ascending=False,
).reset_index(drop=True)


pair_df.to_csv(
    METRICS_DIR / "cnn_all_misclassification_pairs.csv",
    index=False,
)


print("\n" + "=" * 75)
print("TOP 15 MISCLASSIFICATION PAIRS")
print("=" * 75)

print(pair_df.head(15).to_string(index=False))


# ---------------------------------------------------------
# Per-class error rates
# ---------------------------------------------------------
class_rows = []

for class_id, class_name in enumerate(CLASS_NAMES):
    mask = y_test == class_id

    total = mask.sum()

    correct_count = np.sum(predictions[mask] == y_test[mask])

    error_count = total - correct_count

    accuracy = correct_count / total

    error_rate = error_count / total

    avg_confidence = confidence[mask].mean()

    wrong_mask = mask & incorrect

    if wrong_mask.sum() > 0:
        wrong_confidence = confidence[wrong_mask].mean()

    else:
        wrong_confidence = np.nan

    class_rows.append(
        {
            "Class_ID": class_id,
            "Class_Name": class_name,
            "Total": int(total),
            "Correct": int(correct_count),
            "Errors": int(error_count),
            "Accuracy": float(accuracy),
            "Error_Rate": float(error_rate),
            "Average_Confidence": float(avg_confidence),
            "Wrong_Prediction_Confidence": float(wrong_confidence),
        }
    )


class_df = pd.DataFrame(class_rows)

class_df.to_csv(
    METRICS_DIR / "cnn_per_class_error_analysis.csv",
    index=False,
)


print("\n" + "=" * 75)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 75)

display_df = class_df.copy()

display_df["Accuracy"] *= 100
display_df["Error_Rate"] *= 100

print(
    display_df[
        [
            "Class_Name",
            "Correct",
            "Errors",
            "Accuracy",
            "Error_Rate",
        ]
    ].to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2f}%".format,
            "Error_Rate": "{:.2f}%".format,
        },
    )
)


# ---------------------------------------------------------
# Most confident incorrect predictions
# ---------------------------------------------------------
incorrect_indices = np.where(incorrect)[0]

sorted_incorrect = incorrect_indices[np.argsort(confidence[incorrect_indices])[::-1]]


print("\n" + "=" * 75)
print("MOST CONFIDENT INCORRECT PREDICTIONS")
print("=" * 75)

for index in sorted_incorrect[:10]:
    print(
        f"Index {index:<5} | "
        f"True: "
        f"{CLASS_NAMES[y_test[index]]:<12} | "
        f"Predicted: "
        f"{CLASS_NAMES[predictions[index]]:<12} | "
        f"Confidence: "
        f"{confidence[index] * 100:.2f}%"
    )


# ---------------------------------------------------------
# Lowest-confidence correct predictions
# ---------------------------------------------------------
correct_indices = np.where(correct)[0]

uncertain_correct = correct_indices[np.argsort(confidence[correct_indices])]


print("\n" + "=" * 75)
print("LOWEST-CONFIDENCE CORRECT PREDICTIONS")
print("=" * 75)

for index in uncertain_correct[:10]:
    print(
        f"Index {index:<5} | "
        f"Class: "
        f"{CLASS_NAMES[y_test[index]]:<12} | "
        f"Confidence: "
        f"{confidence[index] * 100:.2f}%"
    )


# ---------------------------------------------------------
# Figure 21 — Per-class error rate
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

error_percentages = class_df["Error_Rate"] * 100

bars = plt.bar(
    class_df["Class_Name"],
    error_percentages,
)

plt.title("CNN Error Rate by Fashion-MNIST Class")

plt.ylabel("Error Rate (%)")

plt.xticks(
    rotation=40,
    ha="right",
)

for bar, value in zip(
    bars,
    error_percentages,
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.2,
        f"{value:.1f}%",
        ha="center",
        fontsize=8,
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "21_cnn_per_class_error_rate.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Figure 22 — Top misclassification pairs
# ---------------------------------------------------------
top_pairs = pair_df.head(12).copy()

top_pairs["Pair"] = top_pairs["True_Class"] + " → " + top_pairs["Predicted_Class"]

plt.figure(figsize=(12, 7))

plt.barh(
    top_pairs["Pair"][::-1],
    top_pairs["Count"][::-1],
)

plt.title("Most Frequent CNN Misclassification Pairs")

plt.xlabel("Number of Test Images")

plt.ylabel("True → Predicted")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "22_top_misclassification_pairs.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Figure 23 — Most confident errors
# ---------------------------------------------------------
selected = sorted_incorrect[:20]

fig, axes = plt.subplots(
    4,
    5,
    figsize=(13, 11),
)

for ax, index in zip(
    axes.flat,
    selected,
):
    ax.imshow(
        x_test[index],
        cmap="gray",
    )

    true_name = CLASS_NAMES[y_test[index]]

    pred_name = CLASS_NAMES[predictions[index]]

    conf = confidence[index] * 100

    ax.set_title(
        f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.1f}%",
        fontsize=8,
    )

    ax.axis("off")


plt.suptitle(
    "CNN's Most Confident Incorrect Predictions",
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "23_high_confidence_errors.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Figure 24 — Uncertain correct predictions
# ---------------------------------------------------------
selected = uncertain_correct[:20]

fig, axes = plt.subplots(
    4,
    5,
    figsize=(13, 11),
)

for ax, index in zip(
    axes.flat,
    selected,
):
    ax.imshow(
        x_test[index],
        cmap="gray",
    )

    class_name = CLASS_NAMES[y_test[index]]

    conf = confidence[index] * 100

    ax.set_title(
        f"{class_name}\nConfidence: {conf:.1f}%",
        fontsize=8,
    )

    ax.axis("off")


plt.suptitle(
    "Correct Predictions with Lowest CNN Confidence",
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "24_low_confidence_correct_predictions.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Prediction margin
# ---------------------------------------------------------
sorted_probabilities = np.sort(
    probabilities,
    axis=1,
)

top_probability = sorted_probabilities[:, -1]

second_probability = sorted_probabilities[:, -2]

prediction_margin = top_probability - second_probability


correct_margin = prediction_margin[correct].mean()

incorrect_margin = prediction_margin[incorrect].mean()


print("\n" + "=" * 75)
print("PREDICTION MARGIN ANALYSIS")
print("=" * 75)

print(f"Mean prediction margin (correct)   : {correct_margin:.4f}")

print(f"Mean prediction margin (incorrect) : {incorrect_margin:.4f}")


# ---------------------------------------------------------
# Figure 25 — Prediction margins
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.hist(
    prediction_margin[correct],
    bins=30,
    alpha=0.6,
    label="Correct",
)

plt.hist(
    prediction_margin[incorrect],
    bins=30,
    alpha=0.6,
    label="Incorrect",
)

plt.title("CNN Prediction Margin Distribution")

plt.xlabel("Top Probability − Second Highest Probability")

plt.ylabel("Number of Predictions")

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "25_prediction_margin_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------
error_summary = {
    "total_test_samples": int(len(y_test)),
    "correct_predictions": int(correct.sum()),
    "incorrect_predictions": int(incorrect.sum()),
    "overall_error_rate": float(incorrect.mean()),
    "mean_correct_prediction_margin": float(correct_margin),
    "mean_incorrect_prediction_margin": float(incorrect_margin),
    "hardest_class": class_df.sort_values("Accuracy").iloc[0]["Class_Name"],
    "best_class": class_df.sort_values(
        "Accuracy",
        ascending=False,
    ).iloc[0]["Class_Name"],
    "most_common_error": {
        "true_class": pair_df.iloc[0]["True_Class"],
        "predicted_class": pair_df.iloc[0]["Predicted_Class"],
        "count": int(pair_df.iloc[0]["Count"]),
    },
}


with open(
    METRICS_DIR / "cnn_error_analysis_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        error_summary,
        file,
        indent=4,
    )


print("\n" + "=" * 75)
print("ERROR ANALYSIS FILES GENERATED")
print("=" * 75)

print("\nFigures:")
print("21_cnn_per_class_error_rate.png")
print("22_top_misclassification_pairs.png")
print("23_high_confidence_errors.png")
print("24_low_confidence_correct_predictions.png")
print("25_prediction_margin_distribution.png")

print("\nMetrics:")
print("cnn_all_misclassification_pairs.csv")
print("cnn_per_class_error_analysis.csv")
print("cnn_error_analysis_summary.json")

print("\nDeep CNN error analysis completed successfully.")
