# Fashion-MNIST Image Classification Using Deep Learning

## Project Overview

This project implements a complete deep-learning image-classification workflow using the Fashion-MNIST dataset. The objective is to classify 28×28 grayscale clothing images into 10 fashion categories while comparing a traditional fully connected neural-network baseline with a Convolutional Neural Network (CNN).

The project includes dataset exploration, preprocessing, baseline modelling, CNN development, model evaluation, confidence analysis, calibration analysis, statistical comparison, and detailed error analysis.

Rather than evaluating the models only using overall accuracy, the project investigates class-level performance, recurring misclassification patterns, prediction confidence, calibration, and examples where the CNN corrects errors made by the baseline model.

---

## Dataset

Fashion-MNIST contains 70,000 grayscale images divided into:

- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 pixels
- Pixel range: 0–255
- Number of classes: 10

Each class contains exactly 6,000 training images and 1,000 test images, making the dataset perfectly balanced.

### Classes

| ID | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

No missing values were found in either the training or test images.

---

## Exploratory Data Analysis

The initial exploration examined:

- Dataset dimensions
- Class distribution
- Pixel-value range
- Pixel intensity distribution
- Example images from all classes
- Random image samples
- Mean image representation for each class
- Per-class pixel statistics

The training dataset contains an equal number of samples for every category, so class imbalance treatment was not required.

Pixel values originally ranged from 0 to 255 and were normalized to the range 0–1 before model training.

---

## Baseline Neural Network

A fully connected neural network was developed as the baseline model.

### Architecture

- Flatten layer
- Dense layer — 128 units, ReLU
- Dropout — 30%
- Dense layer — 64 units, ReLU
- Dropout — 20%
- Output layer — 10 units, Softmax

Total trainable parameters: **109,386**

The model used the Adam optimizer and sparse categorical cross-entropy loss.

Early stopping was used to restore the model weights from the epoch with the best validation loss.

### Baseline Results

| Metric | Result |
|---|---:|
| Test Accuracy | **88.11%** |
| Test Loss | **0.3320** |
| Correct Predictions | **8,811** |
| Incorrect Predictions | **1,189** |

The strongest baseline classes included Bag, Trouser, Sandal, and Ankle boot.

The most difficult baseline class was Shirt, with an accuracy of only **70.30%**.

---

## Convolutional Neural Network

A deeper CNN was developed to preserve and learn spatial patterns in the images.

### CNN Architecture

The network contains:

1. Conv2D — 32 filters
2. Batch Normalization
3. Conv2D — 32 filters
4. Max Pooling
5. Dropout
6. Conv2D — 64 filters
7. Batch Normalization
8. Conv2D — 64 filters
9. Max Pooling
10. Dropout
11. Flatten
12. Dense — 128 units
13. Dropout
14. Softmax output — 10 classes

Early stopping and learning-rate reduction were applied during training.

---

## CNN Results

| Metric | Result |
|---|---:|
| Test Accuracy | **93.34%** |
| Correct Predictions | **9,334** |
| Incorrect Predictions | **666** |

The CNN provided a substantial improvement over the dense baseline.

---

## Baseline vs CNN

| Metric | Baseline | CNN |
|---|---:|---:|
| Accuracy | 88.11% | **93.34%** |
| Incorrect Predictions | 1,189 | **666** |
| Mean Confidence | 0.8856 | **0.9535** |
| ECE | **0.0104** | 0.0202 |

The absolute accuracy improvement was:

**+5.23 percentage points**

The CNN eliminated:

**523 net classification errors**

This represents a:

**43.99% relative reduction in errors**

---

## Paired Prediction Analysis

The models were also compared on the exact same test images.

| Prediction Outcome | Images |
|---|---:|
| Both models correct | 8,619 |
| Both models incorrect | 474 |
| CNN corrected baseline error | **715** |
| CNN introduced new error | 192 |

The CNN therefore corrected far more baseline mistakes than the number of new mistakes it introduced.

A paired McNemar exact test produced a highly significant result (**p < 0.001**), providing statistical evidence that the improvement in predictions was not explained simply by random variation on the test examples.

---

## Per-Class Performance Improvement

| Class | Baseline | CNN | Change |
|---|---:|---:|---:|
| T-shirt/top | 82.40% | 88.40% | +6.00 |
| Trouser | 96.70% | 98.20% | +1.50 |
| Pullover | 77.10% | 91.00% | **+13.90** |
| Dress | 86.60% | 93.20% | +6.60 |
| Coat | 82.10% | 92.40% | **+10.30** |
| Sandal | 96.80% | 99.00% | +2.20 |
| Shirt | 70.30% | 78.30% | +8.00 |
| Sneaker | 94.80% | 98.00% | +3.20 |
| Bag | 97.70% | 98.70% | +1.00 |
| Ankle boot | 96.60% | 96.20% | -0.40 |

The largest CNN improvement occurred for **Pullover**, followed by **Coat** and **Shirt**.

Only Ankle boot showed a small reduction of 0.40 percentage points.

---

## Detailed Error Analysis

Although the CNN achieved 93.34% accuracy, performance varied significantly between clothing categories.

### Hardest Class

**Shirt**

- Correct: 783
- Incorrect: 217
- Accuracy: 78.30%
- Error rate: 21.70%

The Shirt category remains difficult because its visual characteristics overlap with T-shirts, pullovers, coats, and dresses.

### Most Frequent Misclassifications

| True Class | Predicted Class | Count |
|---|---|---:|
| T-shirt/top | Shirt | 83 |
| Shirt | T-shirt/top | 83 |
| Shirt | Coat | 61 |
| Shirt | Pullover | 41 |
| Coat | Shirt | 38 |
| Pullover | Shirt | 35 |
| Ankle boot | Sneaker | 34 |
| Pullover | Coat | 34 |
| Shirt | Dress | 28 |
| Dress | Shirt | 26 |

The dominant errors therefore occur mainly between visually similar upper-body clothing categories.

---

## Confidence Analysis

Average CNN confidence:

- Correct predictions: **0.9692**
- Incorrect predictions: **0.7340**

The CNN is therefore generally more confident when it is correct.

However, several incorrect predictions were made with confidence close to 100%. This demonstrates an important limitation: high softmax confidence does not guarantee a correct classification.

The CNN also produced a larger Expected Calibration Error than the baseline:

- Baseline ECE: **0.0104**
- CNN ECE: **0.0202**

Therefore, although the CNN achieves considerably higher classification accuracy, its predicted probabilities are slightly less well calibrated.

---

## Prediction Margin Analysis

The prediction margin was calculated as the difference between the highest and second-highest predicted class probabilities.

Average margin:

- Correct predictions: **0.9423**
- Incorrect predictions: **0.5173**

Correct predictions therefore tend to have a substantially clearer separation between the most likely and second-most-likely classes.

Incorrect predictions show much smaller margins on average, indicating greater ambiguity.

---

## Key Findings

The analysis produced several important findings:

1. CNNs significantly outperform fully connected neural networks for Fashion-MNIST image classification.
2. Test accuracy improved from 88.11% to 93.34%.
3. CNN modelling reduced total classification errors by approximately 44%.
4. Pullover, Coat, and Shirt benefited most from spatial feature learning.
5. Shirt remained the most difficult class.
6. Most errors occur among visually similar upper-body clothing categories.
7. CNN predictions are generally more confident than baseline predictions.
8. Higher accuracy did not result in better probability calibration.
9. Some incorrect CNN predictions were made with extremely high confidence.
10. Prediction margin provides useful additional information about model uncertainty.

---

## Project Structure

```text
Fashion_MNIST_Deep_Learning_Project/
│
├── models/
│   ├── fashion_mnist_baseline.keras
│   └── fashion_mnist_cnn.keras
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
├── report/
│
├── src/
│   ├── explore.py
│   ├── train_baseline.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   └── error_analysis.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- uv

---

## Running the Project

### Clone the repository

```bash
git clone <repository-url>
cd Fashion_MNIST_Deep_Learning_Project
```

### Create the environment

```bash
uv sync
```

### Dataset exploration

```bash
uv run python src/explore.py
```

### Train the baseline neural network

```bash
uv run python src/train_baseline.py
```

### Train the CNN

```bash
uv run python src/train_cnn.py
```

### Compare the models

```bash
uv run python src/evaluate.py
```

### Run detailed CNN error analysis

```bash
uv run python src/error_analysis.py
```

Fashion-MNIST is automatically downloaded through TensorFlow/Keras when required.

---

## Generated Outputs

The project generates more than 20 analytical visualizations, including:

- Class distribution
- Sample clothing images
- Pixel intensity distribution
- Mean images by category
- Baseline confusion matrix
- Baseline learning curves
- CNN confusion matrix
- CNN learning curves
- Correct and incorrect CNN examples
- Confidence distributions
- Baseline-vs-CNN comparison
- Per-class accuracy comparison
- Class-level CNN improvement
- Error-count comparison
- Calibration curves
- CNN-corrected baseline errors
- Per-class CNN error rates
- Frequent misclassification pairs
- High-confidence errors
- Low-confidence correct predictions
- Prediction-margin distribution

Machine-readable CSV and JSON result files are also stored under `outputs/metrics/`.

---

## Conclusion

The project demonstrates that preserving spatial information through convolutional layers provides a clear advantage for image classification.

The baseline neural network achieved 88.11% test accuracy, while the CNN increased performance to 93.34% and reduced errors by approximately 44%.

The deeper evaluation also demonstrated that overall accuracy alone does not fully describe model quality. Class-specific errors, calibration, prediction confidence, and misclassification patterns revealed important model limitations that would not be visible from a single accuracy value.

The final CNN therefore provides a strong Fashion-MNIST classifier while the detailed analysis identifies clear opportunities for further improvement, particularly for visually similar clothing categories such as Shirt, T-shirt/top, Pullover, and Coat.