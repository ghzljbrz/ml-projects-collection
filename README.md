# 🧠 Machine Learning Projects Collection

**A collection of machine learning, deep learning, and reinforcement learning projects — spanning classical ML from first principles, applied signal processing, dimensionality reduction, and deep RL — built as coursework and independent study.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Classical%20ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20RL-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Deep%20Q--Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-From--Scratch%20ML-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
</p>

---

## 📌 About This Collection

Each project in this repository lives in its own folder with a dedicated README, dataset description, methodology, and results. Rather than treating every project as a single library call, most of them **derive the underlying algorithm from first principles** (in NumPy or PyTorch) alongside a library baseline — from Naive Bayes and PCA to Q-Learning and Deep Q-Networks — to demonstrate both the math and the engineering behind classical and modern machine learning.

Across the nine projects here, the recurring themes are:
- 🧮 **Implementing algorithms from scratch** to validate understanding, not just calling `.fit()`
- 🎛️ **Systematic hyperparameter search** (grid, random, and metaheuristic methods like PSO/DE) over ad-hoc tuning
- 📊 **Honest evaluation** — reporting train/test gaps, overfitting checks, and confusion matrices rather than headline accuracy alone
- 🔬 **Domain diversity** — time series, signal processing, NLP, computer vision, tabular data, and reinforcement learning

## 🗂️ Projects

| # | Project | Domain | Key Techniques |
|---|---|---|---|
| 1 | [Collaborative Weather Forecasting](./collaborative-weather-forecasting) | Time Series Regression | From-scratch gradient descent, Linear/Ridge/Lasso, multi-city collaborative features |
| 2 | [Bearing Fault Diagnosis (MaFaulDa)](./bearing-fault-diagnosis-mafaulda) | Signal Processing | Wavelet & FFT feature extraction, hierarchical One-vs-Rest classification |
| 3 | [Spam Detection (Naive Bayes)](./spam-detection-naive-bayes) | NLP / Text Classification | From-scratch Multinomial Naive Bayes vs. scikit-learn |
| 4 | [Digit Recognition (KNN + PCA)](./knn-mnist-digit-recognition) | Computer Vision | K-hyperparameter sweep, PCA dimensionality reduction |
| 5 | [Sales Category Prediction](./decision-tree-sales-prediction) | Tabular / Business Analytics | Hand-computed entropy & information gain, GridSearchCV, overfitting diagnostics |
| 6 | [Air Quality Prediction & Classification](./air-quality-svm-svr) | Time Series / SVM | SVR & SVM kernel comparison, from-scratch SVM, PSO/DE hyperparameter tuning |
| 7 | [Dimensionality Reduction Study](./dimensionality-reduction-pca-lda-tsne) | Computer Vision | From-scratch PCA, PCA denoising, LDA separability analysis, t-SNE |
| 8 | [Wumpus World RL Agent](./wumpus-world-reinforcement-learning) | Reinforcement Learning | Tabular Q-Learning vs. Deep Q-Network |
| 9 | [Lunar Lander DQN/DDQN](./lunar-lander-dqn-ddqn) | Deep Reinforcement Learning | DQN hyperparameter grid search, Double DQN |

## 🛠️ Tech Stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `PyTorch` · `TensorFlow / Keras` · `SciPy` · `PyWavelets` · `cvxopt` · `Gymnasium` · `Matplotlib` · `Seaborn`

## 🚀 Getting Started

Each project is self-contained. Clone the repository, then enter any project folder for its specific setup instructions:

```bash
git clone https://github.com/<your-username>/ml-projects-collection.git
cd ml-projects-collection/<project-folder>
# See that project's README for installation and run instructions
```

## 👥 Authors

- [Ghzal Jabbari](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This collection is available under the MIT License — feel free to explore, fork, and build on it.
