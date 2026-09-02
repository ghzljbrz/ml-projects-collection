# 🌫️ Air Quality (PM2.5) Prediction & Classification with SVM/SVR

**A dual-pipeline project that both predicts continuous PM2.5 pollution levels (regression) and classifies air quality into AQI health categories (classification) — comparing kernel choices, custom from-scratch SVM implementations, and four different hyperparameter-optimization strategies, including metaheuristics.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-SVM%20%2F%20SVR-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="cvxopt" src="https://img.shields.io/badge/cvxopt-Quadratic%20Programming-3776AB?style=for-the-badge">
  <img alt="PSO/DE" src="https://img.shields.io/badge/PSO%20%2F%20DE-Metaheuristic%20Tuning-8A2BE2?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

Air pollution (PM2.5) monitoring is a time series problem with two natural framings: **"how much"** (a continuous pollution level) and **"how bad"** (a health-risk category like Good, Moderate, or Hazardous). This project tackles both, using **Support Vector Machines** as the common modeling thread — first deriving the SVM optimization problem from scratch via quadratic programming, then building two full pipelines:

1. **Support Vector Regression (SVR)** to predict raw PM2.5 concentration.
2. **Support Vector Classification (SVM)** to predict the corresponding AQI health category.

Both pipelines share a rich, time-series-aware feature engineering process (lag features, rolling statistics, cyclic time encoding, seasonal indicators), and both go well beyond a single default model — comparing kernels, comparing from-scratch vs. library implementations, and comparing four distinct hyperparameter-search strategies.

The core question:

> Which kernel and hyperparameter-search strategy best models the highly non-linear relationship between weather/temporal features and air quality — and how does a from-scratch SVM compare to scikit-learn's implementation?

## ✨ What This Project Demonstrates

- 🧮 **SVM theory from first principles** — deriving and solving the SVM dual optimization problem directly via quadratic programming (`cvxopt`), before ever calling `sklearn.svm`.
- 🛠️ **Custom SVM implementations** — a from-scratch SGD-based multi-class SVM (One-vs-Rest) and a **Random Fourier Features** approximation of the RBF kernel, built entirely in NumPy.
- 🕰️ **Time-series feature engineering** — lag features (1h, 3h, 24h, 168h), rolling statistics, cyclic (sine/cosine) encoding of hour-of-day, and seasonal indicators derived from raw timestamped sensor data.
- 🧹 **Rigorous preprocessing** — missing-value imputation, categorical encoding, outlier handling, and class-imbalance correction via undersampling for the classification target.
- 🔬 **Kernel comparison, done properly** — Linear, Polynomial, and RBF kernels benchmarked head-to-head for both the regression (SVR) and classification (SVM) tasks.
- 🎛️ **Four hyperparameter-optimization strategies compared** — `GridSearchCV`, `RandomizedSearchCV`, **Particle Swarm Optimization (PSO)**, and **Differential Evolution (DE)** — going well beyond standard grid search to explore metaheuristic tuning.
- 📊 **Proper train/validation/test methodology** — a genuine three-way split (70/15/15) with the validation set used for model selection and the test set reserved purely for final evaluation.

## 🗂️ Dataset

- **Domain:** Hourly air-quality and meteorological sensor data — PM2.5 concentration plus weather variables (`DEWP`, `TEMP`, `PRES`, `Iws`, `cbwd`, `snow`, `rain`) and timestamp fields (`year`, `month`, `day`, `hour`).
- **Regression target:** raw `pm2.5` concentration (µg/m³).
- **Classification target:** `AQI Category` — a multi-class health-risk label (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous) derived from standard AQI breakpoints for PM2.5.
- **Preprocessing:** mean imputation for missing PM2.5 values, categorical encoding, outlier treatment, class-balanced undersampling (classification only), and `StandardScaler` normalization.

## 🧪 Methodology

| Stage | Description |
|---|---|
| **1. QP foundations** | Solve a toy SVM dual problem directly via quadratic programming to validate the underlying theory. |
| **2. Preprocessing** | Impute missing values, encode categoricals, detect and treat outliers, derive the AQI category target. |
| **3. Feature engineering** | Build lag features, rolling statistics, cyclic time encodings, and seasonal indicators; explore correlations via heatmaps and pair plots. |
| **4. Balancing (classification only)** | Undersample the majority class(es) to address AQI category imbalance. |
| **5. Normalization & splitting** | Standard-scale features and split into train/validation/test (70/15/15). |
| **6. Kernel comparison** | Train and evaluate Linear, Polynomial, and RBF kernels for both SVR and SVM. |
| **7. From-scratch models** | Implement and evaluate a custom SGD-based SVM and a Random-Fourier-Features RBF-SVM. |
| **8. Hyperparameter search** | Tune `C` and `gamma` via Grid Search, Random Search, PSO, and Differential Evolution. |
| **9. Final evaluation** | Compare all approaches on the held-out test set via accuracy/R², classification reports, and confusion matrices. |

## 🤖 Notebooks

### 1. PM2.5 Regression — `svr_pm25_regression.ipynb`
Predicts continuous PM2.5 concentration using Support Vector Regression, comparing **Linear**, **Polynomial (degree 3)**, and **RBF** kernels on MAE, MSE, RMSE, and R².

### 2. AQI Category Classification — `svm_aqi_classification.ipynb`
The more extensive pipeline: classifies hourly readings into one of six AQI health categories using SVM, including:
- Kernel comparison (Linear, Polynomial, RBF) with validation-set evaluation and confusion matrices
- From-scratch **SGD multi-class SVM (OvR)** and **Random Fourier Features RBF-SVM**
- Hyperparameter tuning via **Grid Search**, **Random Search**, **Particle Swarm Optimization**, and **Differential Evolution**

## 📈 Results Snapshot

**SVR kernel comparison (PM2.5 regression):**

| Kernel | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 0.262 | 0.453 | 0.749 |
| Polynomial (deg. 3) | 0.346 | 0.499 | 0.696 |
| **RBF** | 0.292 | **0.407** | **0.797** (best) |

**SVM kernel comparison (AQI classification, validation accuracy):**

| Kernel | Accuracy |
|---|---|
| Linear | 72.4% |
| Polynomial | 74.8% |
| RBF | 78.6% |

**Hyperparameter-optimization strategy comparison (test/validation accuracy):**

| Strategy | Kernel | Accuracy |
|---|---|---|
| Grid Search | Linear | **98.0%** |
| Random Search | Linear | **98.1%** (best overall) |
| Grid Search | RBF | 91.8% |
| Random Search | RBF | 71.4% |
| Particle Swarm Optimization | RBF | 93.4% |
| Differential Evolution | RBF | 93.5% |
| From-scratch SGD SVM | Linear (OvR) | 37.8% |

> 📌 **Key finding:** properly tuned SVMs dramatically outperform untuned defaults — the tuned **Linear kernel reaches ~98% accuracy** (vs. 72% untuned), showing that hyperparameter search mattered far more than kernel choice for this dataset. Among RBF-kernel tuning strategies, the metaheuristic methods (PSO, DE) **outperformed both Grid and Random Search**, while the from-scratch SGD SVM — a from-first-principles reference implementation rather than a production model — lagged well behind, highlighting the real-world value of scikit-learn's optimized solvers and the tuning strategies explored here.

## 🗃️ Project Structure

```
.
├── svr_pm25_regression.ipynb       # SVR kernel comparison for continuous PM2.5 prediction
├── svm_aqi_classification.ipynb    # SVM classification, from-scratch models, and 4-way hyperparameter search
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone [https://github.com/ghzljbrz/ml-projects-collection].git
cd ml-projects-collection

# Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn cvxopt pyswarm gdown
```

> Both notebooks were developed in Google Colab and fetch the dataset directly via Google Drive; no manual download is required.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `SciPy` · `cvxopt` · `PySwarm (PSO)` · `Matplotlib` · `Seaborn`

## 🔮 Future Improvements

- Extend the Random Fourier Features SVM with a proper multi-class decision rule and benchmark it directly against the SGD SVM.
- Apply the same PSO/DE hyperparameter search used for the RBF classifier to the SVR regression task.
- Add k-fold cross-validation on the final chosen model for a more robust test-set accuracy estimate.
- Package the shared time-series feature-engineering logic (lag features, rolling stats, cyclic encoding) into a reusable module used by both notebooks.

## 👥 Authors

- [Ghzal Jabbari](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.
