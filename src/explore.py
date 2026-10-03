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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Fashion-MNIST class names
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
print("=" * 70)
print("FASHION-MNIST DATASET EXPLORATION")
print("=" * 70)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("\nDataset successfully loaded.")

print("\nTraining image shape:")
print(x_train.shape)

print("\nTraining label shape:")
print(y_train.shape)

print("\nTest image shape:")
print(x_test.shape)

print("\nTest label shape:")
print(y_test.shape)


# ---------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Number of training images : {len(x_train)}")
print(f"Number of test images     : {len(x_test)}")
print(f"Total images              : {len(x_train) + len(x_test)}")

print(f"\nImage height : {x_train.shape[1]}")
print(f"Image width  : {x_train.shape[2]}")

print(f"\nMinimum pixel value : {x_train.min()}")
print(f"Maximum pixel value : {x_train.max()}")
print(f"Mean pixel value    : {x_train.mean():.4f}")
print(f"Standard deviation  : {x_train.std():.4f}")

print(f"\nNumber of classes : {len(np.unique(y_train))}")


# ---------------------------------------------------------
# Check missing / invalid values
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

print(f"Missing values in training images: {np.isnan(x_train).sum()}")
print(f"Missing values in test images    : {np.isnan(x_test).sum()}")

print(f"\nUnique training labels: {np.unique(y_train)}")
print(f"Unique test labels    : {np.unique(y_test)}")


# ---------------------------------------------------------
# Class distribution
# ---------------------------------------------------------
train_counts = pd.Series(y_train).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

distribution_df = pd.DataFrame(
    {
        "Class_ID": range(10),
        "Class_Name": CLASS_NAMES,
        "Train_Count": train_counts.values,
        "Test_Count": test_counts.values,
    }
)

distribution_df["Total_Count"] = (
    distribution_df["Train_Count"] + distribution_df["Test_Count"]
)

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(distribution_df.to_string(index=False))

distribution_df.to_csv(METRICS_DIR / "class_distribution.csv", index=False)


# ---------------------------------------------------------
# Figure 1 — Class distribution
# ---------------------------------------------------------
plt.figure(figsize=(11, 6))

sns.barplot(
    data=distribution_df,
    x="Class_Name",
    y="Train_Count",
)

plt.title("Fashion-MNIST Training Class Distribution")
plt.xlabel("Clothing Category")
plt.ylabel("Number of Training Images")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(FIGURE_DIR / "01_class_distribution.png", dpi=300, bbox_inches="tight")

plt.close()


# ---------------------------------------------------------
# Figure 2 — One example from every class
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for class_id in range(10):
    index = np.where(y_train == class_id)[0][0]

    ax = axes.flat[class_id]

    ax.imshow(x_train[index], cmap="gray")

    ax.set_title(f"{class_id}: {CLASS_NAMES[class_id]}")

    ax.axis("off")

plt.suptitle("Example Image from Each Fashion-MNIST Class", fontsize=14)

plt.tight_layout()

plt.savefig(FIGURE_DIR / "02_class_examples.png", dpi=300, bbox_inches="tight")

plt.close()


# ---------------------------------------------------------
# Figure 3 — Random training samples
# ---------------------------------------------------------
rng = np.random.default_rng(seed=42)

random_indices = rng.choice(len(x_train), size=25, replace=False)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for ax, index in zip(axes.flat, random_indices):
    ax.imshow(x_train[index], cmap="gray")

    ax.set_title(CLASS_NAMES[y_train[index]], fontsize=8)

    ax.axis("off")

plt.suptitle("Random Fashion-MNIST Training Samples", fontsize=14)

plt.tight_layout()

plt.savefig(FIGURE_DIR / "03_random_samples.png", dpi=300, bbox_inches="tight")

plt.close()


# ---------------------------------------------------------
# Figure 4 — Pixel intensity distribution
# ---------------------------------------------------------
sample_pixels = x_train[:5000].reshape(-1)

plt.figure(figsize=(10, 6))

plt.hist(
    sample_pixels,
    bins=50,
)

plt.title("Pixel Intensity Distribution")

plt.xlabel("Pixel Intensity")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(FIGURE_DIR / "04_pixel_distribution.png", dpi=300, bbox_inches="tight")

plt.close()


# ---------------------------------------------------------
# Per-class pixel statistics
# ---------------------------------------------------------
class_statistics = []

for class_id, class_name in enumerate(CLASS_NAMES):
    class_images = x_train[y_train == class_id]

    class_statistics.append(
        {
            "Class_ID": class_id,
            "Class_Name": class_name,
            "Image_Count": len(class_images),
            "Mean_Pixel_Value": class_images.mean(),
            "Pixel_Std": class_images.std(),
        }
    )

class_statistics_df = pd.DataFrame(class_statistics)

print("\n" + "=" * 70)
print("PER-CLASS PIXEL STATISTICS")
print("=" * 70)

print(class_statistics_df.to_string(index=False))

class_statistics_df.to_csv(METRICS_DIR / "class_pixel_statistics.csv", index=False)


# ---------------------------------------------------------
# Figure 5 — Mean image for every class
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for class_id in range(10):
    class_images = x_train[y_train == class_id]

    mean_image = class_images.mean(axis=0)

    ax = axes.flat[class_id]

    ax.imshow(mean_image, cmap="gray")

    ax.set_title(CLASS_NAMES[class_id])

    ax.axis("off")

plt.suptitle("Average Image Representation for Each Class", fontsize=14)

plt.tight_layout()

plt.savefig(FIGURE_DIR / "05_mean_class_images.png", dpi=300, bbox_inches="tight")

plt.close()


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------
summary = {
    "training_samples": len(x_train),
    "test_samples": len(x_test),
    "total_samples": len(x_train) + len(x_test),
    "image_height": x_train.shape[1],
    "image_width": x_train.shape[2],
    "number_of_classes": len(np.unique(y_train)),
    "minimum_pixel_value": int(x_train.min()),
    "maximum_pixel_value": int(x_train.max()),
    "mean_pixel_value": float(x_train.mean()),
    "pixel_standard_deviation": float(x_train.std()),
}

summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])

summary_df.to_csv(METRICS_DIR / "dataset_summary.csv", index=False)


print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print("Figures:")
print("01_class_distribution.png")
print("02_class_examples.png")
print("03_random_samples.png")
print("04_pixel_distribution.png")
print("05_mean_class_images.png")

print("\nMetrics:")
print("dataset_summary.csv")
print("class_distribution.csv")
print("class_pixel_statistics.csv")

print("\nFashion-MNIST exploration completed successfully.")
