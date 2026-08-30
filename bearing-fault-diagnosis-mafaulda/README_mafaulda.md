# ⚙️ Bearing Fault Diagnosis with Wavelet & FFT Feature Extraction

**A hierarchical machine learning pipeline that detects and classifies 10 types of rotating-machinery faults from raw vibration sensor data, using the MaFaulDa benchmark dataset.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Classification-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="SciPy" src="https://img.shields.io/badge/SciPy-Signal%20Processing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white">
  <img alt="PyWavelets" src="https://img.shields.io/badge/PyWavelets-Wavelet%20Transform-4B8BBE?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

Rotating machinery (motors, pumps, turbines) accounts for a large share of industrial equipment failures, and early fault detection from vibration signals is a core problem in predictive maintenance. This project builds a **hierarchical fault-diagnosis system**: it first separates normal operation from faulty operation, then narrows down the specific fault category, using accelerometer signals from the **MaFaulDa (Machinery Fault Database)** benchmark.

The pipeline compares two complementary signal-processing strategies for turning raw, high-frequency vibration signals into machine-learning-ready features:

> Which representation captures fault signatures better — time-domain wavelet decomposition, or frequency-domain FFT analysis — and how well does each generalize across ten distinct fault types?

## ✨ What This Project Demonstrates

- 📡 **Raw signal engineering** — cleaning and standardizing multi-channel accelerometer/tachometer time series from a real industrial fault simulator.
- 🌊 **Wavelet-domain feature extraction** — multi-level discrete wavelet decomposition (`PyWavelets`) producing statistical descriptors (mean, std, variance, RMS, kurtosis, skewness, zero-crossing rate, entropy) per approximation/detail coefficient, plus Hilbert-transform envelope features.
- 📈 **Frequency-domain feature extraction** — FFT-based spectral features as an alternative representation, evaluated head-to-head against the wavelet approach.
- 🪟 **Sliding-window segmentation** — chunking long raw signals into fixed-size, overlapping windows to generate a large, balanced feature dataset from a limited number of raw recordings.
- 🧭 **Hierarchical / One-vs-Rest classification** — a cascade of binary and multi-class Logistic Regression classifiers: normal vs. faulty, then fault family vs. fault family, down to each of the 10 individual fault classes.
- 📊 **Rigorous evaluation** — confusion matrices and full classification reports (precision/recall/F1) for every binary and multi-class stage.

## 🗂️ Dataset

- **Name:** [MaFaulDa — Machinery Fault Database](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html), created by the Signal, Multimedia and Telecommunications (SMT) Lab, Federal University of Rio de Janeiro (UFRJ).
- **Acquisition setup:** SpectraQuest Alignment-Balance-Vibration Trainer (ABVT) Machinery Fault Simulator, with triaxial accelerometers on the underhang and overhang bearings, a tachometer, and a microphone, sampled at 50 kHz.
- **Classes used (10):** normal, imbalance, horizontal misalignment, vertical misalignment, overhang (ball / cage / outer race) fault, underhang (ball / cage / outer race) fault.
- **Preprocessing:** irrelevant/low-information channels dropped, signals standardized (z-score), then segmented via a sliding window (window size 2000, stride 500) before feature extraction.

## 🧪 Methodology

| Stage | Description |
|---|---|
| **1. Preprocessing** | Load raw per-fault CSVs, drop uninformative channels, standardize all numerical sensor columns. |
| **2. Segmentation** | Apply a sliding window across each long signal to generate many fixed-length samples per fault class. |
| **3. Feature extraction (Wavelet)** | Decompose each window with a discrete wavelet transform (`bior3.1`, 4 levels); compute statistical + entropy + Hilbert-envelope features per channel. |
| **4. Feature extraction (FFT)** | Alternatively, compute FFT-based spectral features per window for comparison. |
| **5. Classification** | Train One-vs-Rest Logistic Regression classifiers: multi-class (A–J), binary (normal vs. faulty), and per-fault (one fault vs. all others). |
| **6. Evaluation** | Score every stage with accuracy, confusion matrices, and classification reports. |

## 🤖 Notebooks

### 1. Signal Preprocessing — `01_preprocessing.ipynb`
Loads a raw vibration CSV, visualizes all sensor channels, removes low-information/irrelevant columns, and standardizes the remaining channels — establishing the cleaning pipeline reused across all fault files.

### 2. Feature Extraction & Classification — `02_feature_extraction_classification.ipynb`
The core of the project:
- Downloads all 10 fault-class CSVs, applies sliding-window segmentation, and extracts **wavelet-domain features** (mean, std, var, RMS, kurtosis, skewness, entropy per coefficient band, plus Hilbert-envelope stats).
- Trains hierarchical One-vs-Rest Logistic Regression classifiers on the combined feature set.
- Repeats the pipeline with **FFT-domain features** for direct comparison.
- Reports full classification metrics and confusion matrices at every decision level.

## 📈 Results Snapshot

| Classification Task | Feature Domain | Accuracy |
|---|---|---|
| Multi-class (all 10 faults, A–J) | Wavelet | **100%** |
| Normal vs. Faulty (binary) | Wavelet | **100%** |
| Each individual fault vs. rest | Wavelet | **99.0% – 100%** |
| Normal vs. Other | FFT | 89.0% |
| Horizontal / Vertical Misalignment vs. Other | FFT | 92.5% – 93.2% |
| Overhang / Underhang fault vs. Other | FFT | 97.9% – 100% |

> 📌 **Key finding:** wavelet-domain features consistently outperform FFT-domain features across nearly every fault category, suggesting the transient, non-stationary nature of these fault signatures is better captured in the time-frequency (wavelet) domain than in a purely frequency-domain (FFT) representation.

## 🗃️ Project Structure

```
.
├── 01_preprocessing.ipynb                      # Signal loading, cleaning, standardization
├── 02_feature_extraction_classification.ipynb  # Wavelet & FFT features + hierarchical classification
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn gdown PyWavelets
```

> Both notebooks were developed in Google Colab and fetch data directly from Google Drive via `gdown`; no manual dataset download is required.

## 🛠️ Tech Stack

`Python` · `NumPy` · `Pandas` · `SciPy` · `PyWavelets` · `scikit-learn` · `Matplotlib` · `Seaborn`

## 🔮 Future Improvements

- Move beyond Logistic Regression to non-linear classifiers (e.g. Random Forest, Gradient Boosting, or a small neural network) to test whether the near-perfect wavelet-domain accuracy holds on harder train/test splits.
- Add cross-validation instead of a single train/test split for more robust accuracy estimates.
- Explore combining wavelet and FFT features into a single hybrid feature set.
- Package the preprocessing → feature extraction → classification pipeline into reusable Python modules with a CLI.

## 👥 Authors

- [Ghazal Jabbri](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.

## 🙏 Acknowledgements

Dataset courtesy of the SMT Lab, Federal University of Rio de Janeiro (UFRJ) — [MaFaulDa: Machinery Fault Database](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html).
