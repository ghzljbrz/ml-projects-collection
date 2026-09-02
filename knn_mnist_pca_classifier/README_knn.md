# 🔢 Handwritten Digit Recognition with KNN & PCA (MNIST)

**A K-Nearest Neighbors classifier for handwritten digit recognition on MNIST, with a systematic study of the K hyperparameter and PCA-based dimensionality reduction.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-KNN%20%7C%20PCA-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

This project classifies handwritten digits (0–9) from the **MNIST** dataset using **K-Nearest Neighbors (KNN)** — a simple, distance-based classifier that is highly sensitive to feature scaling, dimensionality, and the choice of `k`. The project doesn't stop at a single trained model; it runs a systematic hyperparameter study and then tackles KNN's best-known weakness — poor scalability with high-dimensional data — using **Principal Component Analysis (PCA)**.

The core questions this project investigates:

> How does classification accuracy change as `k` varies, and can dimensionality reduction via PCA improve both accuracy and efficiency on high-dimensional image data?

## ✨ What This Project Demonstrates

- 🖼️ **Image data as tabular features** — treating each 28×28 pixel MNIST image as a 784-dimensional feature vector for a classical ML pipeline.
- ⚖️ **Feature scaling methodology** — evaluating and justifying the choice between `StandardScaler` and `MinMaxScaler` for distance-based models, based on the data's characteristics.
- 🔍 **Systematic hyperparameter tuning** — sweeping `k` across a wide range (1 to 25) and visualizing the full accuracy-vs-k curve, not just spot-checking a few values.
- 📉 **Dimensionality reduction with PCA** — reducing 784 pixel features down to a small set of principal components, then re-evaluating KNN's accuracy at each reduced dimensionality (20 to 100 components).
- 📊 **Data-driven model selection** — choosing the best `k` and best number of PCA components empirically, backed by accuracy curves rather than guesswork.

## 🗂️ Dataset

- **Source:** MNIST handwritten digit dataset (train + test CSVs combined).
- **Scope used:** First **10,000 samples** after combining and shuffling, each a 28×28 grayscale image (784 pixel features) with a digit label (0–9).
- **Preprocessing:** Min-Max normalization of pixel values to `[0, 1]`.
- **Split:** 70% train / 30% test.

## 🧪 Methodology

| Stage | Description |
|---|---|
| **1. Data loading** | Combine the MNIST train and test CSVs, then subset to the first 10,000 samples. |
| **2. Normalization** | Scale all 784 pixel features with `MinMaxScaler` (chosen over `StandardScaler` to preserve MNIST's bounded, non-negative pixel-intensity structure). |
| **3. Train/test split** | 70/30 split on the normalized feature matrix. |
| **4. Baseline KNN** | Train and evaluate KNN at `k = 3, 5, 9`. |
| **5. K sweep** | Train and evaluate KNN for every odd `k` from 1 to 25, plotting accuracy vs. `k`. |
| **6. PCA + KNN** | Reduce dimensionality with PCA and retrain the best-performing KNN model on the reduced feature space. |
| **7. PCA component sweep** | Repeat PCA + KNN across component counts of 20, 40, 60, 80, and 100, plotting accuracy vs. number of components. |

## 🤖 Notebook

### KNN Digit Classifier with PCA — `knn_mnist_pca_classifier.ipynb`
Implements the full pipeline described above: data loading and normalization, a baseline KNN model, a full hyperparameter sweep over `k`, and a PCA-based dimensionality-reduction study to test whether fewer, more informative features can match or beat full-resolution KNN performance.

## 📈 Results Snapshot

**Effect of `k` (full 784-pixel feature space):**

| k | Accuracy |
|---|---|
| 1 | 94.53% |
| 3 | 94.43% |
| 5 | 94.17% |
| 9 | 93.83% |
| 15 | 93.17% |
| 25 | 92.03% |

> 📌 Accuracy **decreases monotonically** as `k` increases — smaller neighborhoods capture MNIST's fine-grained local digit structure better than larger, smoother decision boundaries.

**Effect of PCA dimensionality (KNN with k = 9):**

| PCA Components | Accuracy |
|---|---|
| 20 | 94.13% |
| 40 | 94.70% |
| **60** | **94.73%** (best) |
| 80 | 94.47% |
| 100 | 94.27% |
| 784 (no PCA) | 91.50% |

> 📌 **Key finding:** PCA doesn't just speed up KNN — it actually **improves accuracy**, with 60 components (a **13x reduction** from 784 raw pixels) outperforming the full-resolution feature space (94.73% vs. 91.50%). This demonstrates that much of the raw pixel data is redundant/noisy for distance-based classification, and that a compact, well-chosen feature representation beats "more features."

## 🗃️ Project Structure

```
.
├── knn_mnist_pca_classifier.ipynb   # KNN digit classification, k-sweep, and PCA dimensionality study
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone [https://github.com/ghzljbrz/ml-projects-collection].git
cd ml-projects-collection

# Install dependencies
pip install pandas numpy scikit-learn matplotlib gdown
```

> The notebook was developed in Google Colab and fetches the MNIST train/test CSVs directly via `gdown`; no manual download is required.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib`

## 🔮 Future Improvements

- Add cross-validation for `k` selection instead of a single train/test split.
- Compare additional distance metrics (Manhattan, cosine) beyond the default Euclidean distance.
- Benchmark KNN + PCA against other classifiers (SVM, Random Forest, a small CNN) on the same data for a broader performance comparison.
- Report training/inference time alongside accuracy to quantify PCA's efficiency gains, not just its accuracy gains.

## 👥 Authors

- [Ghzal Jabbari](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.
