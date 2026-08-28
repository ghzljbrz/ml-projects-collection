# 🌦️ Collaborative Weather Forecasting with Regression Models

**A real-time, collaborative machine learning system that forecasts city temperature using live weather data from multiple neighboring locations.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-from--scratch%20SGD-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Regression-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

This project explores **collaborative machine learning** for weather forecasting: instead of predicting a city's temperature from its own history alone, the model learns from **multiple correlated predictor locations at once**, using humidity, wind speed, and pressure readings from neighboring cities as auxiliary signals.

The core idea (and question the project sets out to answer) is:

> Can we predict a city's temperature at time *t+1* more accurately by collaboratively leveraging weather data from *nearby* cities, instead of relying on that city's data in isolation?

To answer this, the project implements and compares **three different regression strategies** — ranging from a fully hand-built gradient-descent model to scikit-learn's regularized linear models — evaluated on a real 2009 European weather dataset.

## ✨ What This Project Demonstrates

- 🧠 **ML fundamentals from first principles** — a linear/polynomial regression model built from scratch in NumPy, including forward pass, loss computation, manual weight updates, and stochastic gradient descent (no `sklearn.fit()` shortcuts).
- 🔁 **Sliding-window (walk-forward) modeling** — retraining the model over a rolling time window to simulate real-time, streaming forecasts rather than a single static train/test split.
- 🌍 **Multi-source ("collaborative") feature engineering** — combining humidity, wind speed, pressure, and temperature signals from three French weather stations (Tours, Montélimar, Perpignan) as joint predictors.
- ⚖️ **Model benchmarking** — comparing plain Linear Regression against **Ridge** and **Lasso** regularization to study bias/variance trade-offs and coefficient behavior.
- 📊 **Training diagnostics** — tracking and visualizing train/test loss curves across 1,000 training epochs to monitor convergence and detect over/underfitting.
- 🧹 **End-to-end data pipeline** — automated dataset retrieval, cleaning, datetime parsing, feature selection, and min–max normalization.

## 🗂️ Dataset

- **Source:** European weather dataset (2000–2009 daily observations across multiple cities), retrieved programmatically via `gdown`.
- **Scope used:** Filtered to **2009** and to three French cities — **Tours, Montélimar, Perpignan** — with features including humidity, wind speed, pressure, and mean temperature.
- **Preprocessing:** date parsing, column filtering, and min–max normalization of all numerical features.

## 🧪 Methodology

| Stage | Description |
|---|---|
| **1. Ingestion** | Download the raw CSV, parse dates, and subset to the target cities/year. |
| **2. Normalization** | Min–max scale every numeric feature to `[0, 1]`. |
| **3. Feature construction** | For a target city A, combine humidity/wind/pressure + temperature of A with two collaborating cities (B, C). |
| **4. Windowed training** | Slide a fixed-size window across the time series, retraining on each window and predicting the next unseen point. |
| **5. Model training** | Fit each of the three modeling approaches below and log per-epoch train/test loss. |
| **6. Evaluation** | Compare models via Mean Squared Error (MSE) and loss-curve convergence. |

## 🤖 Models Implemented

### 1. Windowed Linear Regression — `01_windowed_linear_regression.ipynb`
A rolling-window approach using `sklearn.linear_model.LinearRegression`, retrained on each window of the time series to produce a one-step-ahead forecast — simulating an online/real-time prediction setting.

### 2. From-Scratch Regression via Gradient Descent — `02_regression_from_scratch_sgd.ipynb`
A custom `polynomialRegression` class implemented purely in **NumPy**, featuring:
- Manual forward pass and MSE loss computation
- Per-sample stochastic weight updates
- Configurable learning rate, epochs, and early-stopping tolerance
- Live progress tracking via `tqdm`
- Train vs. test loss curve visualization across three targets: **temperature, wind speed, and pressure**

Trained over 1,000 epochs, the model's test loss converges from ~4.8 down to **~0.008**, showing stable and consistent learning.

### 3. Regularized Linear Models — `03_linear_ridge_lasso_comparison.ipynb`
A head-to-head comparison of **Linear, Ridge (α=1.0), and Lasso (α=0.1)** regression using `scikit-learn`, with MSE scoring and scatter plots of predicted vs. actual values to visually assess fit quality.

## 📈 Results Snapshot

| Model | Test MSE (2009 French cities data) |
|---|---|
| Linear Regression | ≈ 0.0000 (near-perfect fit on this feature set) |
| Ridge Regression (α=1.0) | ≈ 0.0007 |
| Lasso Regression (α=0.1) | ≈ 0.0544 |
| Custom Gradient-Descent Model | Test loss ↓ from 4.82 → 0.008 over 1,000 epochs |

> ⚠️ Note: The near-zero Linear Regression MSE reflects strong feature collinearity in this particular window/feature setup (see [Future Improvements](#-future-improvements)) — an interesting diagnostic finding in its own right, discussed further in the notebooks.

## 🗃️ Project Structure

```
.
├── 01_windowed_linear_regression.ipynb      # Windowed linear regression (real-time style forecasting)
├── 02_regression_from_scratch_sgd.ipynb     # From-scratch NumPy regression with gradient descent
├── 03_linear_ridge_lasso_comparison.ipynb   # Linear vs. Ridge vs. Lasso comparison
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib tqdm gdown

# Launch Jupyter and run any notebook
jupyter notebook
```

Each notebook downloads the dataset automatically via `gdown` on first run.

## 🛠️ Tech Stack

`Python` · `NumPy` · `Pandas` · `SciPy` · `scikit-learn` · `Matplotlib` · `tqdm` · `Jupyter`

## 🔮 Future Improvements

- Refactor the three notebooks into a single reusable Python package/module
- Add cross-validation and hyperparameter tuning (grid search over α, learning rate, window size)
- Extend the from-scratch model to true polynomial feature expansion
- Investigate feature collinearity to better interpret the near-zero linear regression MSE
- Package as a lightweight API for real-time forecast serving

## 👤 Author

Built as part of a machine learning coursework project on collaborative, multi-location weather forecasting.

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.
