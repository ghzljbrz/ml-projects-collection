# 🌳 Product Sales Category Prediction with Decision Trees

**An interpretable Decision Tree pipeline that predicts a product's sales performance tier (Low / Average / High) from pricing, advertising, and store-placement features — with information-theoretic feature analysis and hyperparameter tuning.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Decision%20Tree-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img alt="Seaborn" src="https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

Retailers need to understand **which product and store-level features actually drive sales** in order to prioritize pricing, advertising, and shelf-placement decisions. This project builds an **interpretable Decision Tree classifier** on a real product-sales dataset, framing the problem not just as "predict sales" but as *"which features carry the most decision-making value, and how much does the tree actually rely on each one?"*

Rather than treating the Decision Tree as a black box, the project derives its splitting criteria from first principles — computing **entropy** and **information gain** by hand — before handing tuning over to scikit-learn's `GridSearchCV`.

The core question:

> Which product, pricing, and store-context features are most predictive of a product's sales tier, and how well can a tuned, interpretable Decision Tree capture that relationship?

## ✨ What This Project Demonstrates

- 🧮 **Information theory from first principles** — custom `calculate_entropy()` and `info_gain()` functions implementing Shannon entropy and information gain directly from their formulas, used to rank every feature's predictive value *before* any model is trained.
- 🏷️ **Thoughtful target engineering** — converting a continuous `Sales` variable into a meaningful 3-class target (`Low` / `Average` / `High`) using quantile-based thresholds, turning a regression-shaped problem into an interpretable classification task.
- 🧹 **End-to-end tabular preprocessing** — missing-value and duplicate checks, categorical encoding (binary and ordinal) for mixed-type business data.
- 🔗 **Exploratory correlation analysis** — a full feature correlation matrix visualized as a heatmap to understand relationships before modeling.
- 🎛️ **Systematic hyperparameter search** — `GridSearchCV` over tree depth, minimum split size, and minimum leaf size (36 candidate configurations, 5-fold CV) rather than manual tuning.
- 🩺 **Overfitting diagnostics** — explicit train-vs-test accuracy comparison with a programmatic over/underfitting check, plus a visualized, pruned decision tree and a full confusion-matrix breakdown (TP/FP/FN/TN per class).

## 🗂️ Dataset

- **Domain:** Retail product sales data (`Company_Data.csv`) — pricing, advertising spend, store demographics, shelf location, and sales for a product across multiple store locations.
- **Features:** `CompPrice`, `Income`, `Advertising`, `Population`, `Price`, `ShelveLoc`, `Age`, `Education`, `Urban`, `US`.
- **Target:** `Sales` (continuous), re-encoded into a 3-class `sales_category` — **Low**, **Average**, **High** — using the 25th/75th percentile as class boundaries.
- **Preprocessing:** missing-value and duplicate checks, binary encoding (`Urban`, `US`), ordinal encoding (`ShelveLoc`: Bad/Medium/Good → 0/1/2).

## 🧪 Methodology

| Stage | Description |
|---|---|
| **1. Loading & cleaning** | Load the CSV, check for missing values and duplicates. |
| **2. Encoding** | Encode categorical features (binary + ordinal) into numeric form. |
| **3. Target engineering** | Bucket the continuous `Sales` target into Low/Average/High via quantile thresholds. |
| **4. Correlation analysis** | Visualize a full feature correlation matrix as a heatmap. |
| **5. Entropy & information gain** | Compute the entropy of the target distribution and the information gain of each feature by hand, ranking features by predictive value. |
| **6. Hyperparameter search** | Run `GridSearchCV` (5-fold CV) over tree depth, min-samples-split, and min-samples-leaf. |
| **7. Evaluation** | Compare train vs. test accuracy, visualize the pruned tree, and report a full classification report + confusion matrix. |

## 🤖 Notebook

### Decision Tree Sales Classifier — `decision_tree_sales_classifier.ipynb`
Implements the complete pipeline above: data cleaning and encoding, quantile-based target creation, hand-computed entropy/information-gain feature ranking, a grid-searched and pruned `DecisionTreeClassifier`, tree visualization, and a full evaluation with an explicit overfitting check.

## 📈 Results Snapshot

**Information gain by feature (relative predictive value, higher = more informative):**

| Feature | Information Gain |
|---|---|
| **Population** | **1.098** (highest) |
| Price | 0.505 |
| Income | 0.386 |
| CompPrice | 0.276 |
| Age | 0.274 |
| ShelveLoc | 0.217 |
| Advertising | 0.171 |
| US | 0.039 |
| Education | 0.039 |
| Urban | 0.0004 (lowest) |

**Tuned Decision Tree performance:**

| Metric | Value |
|---|---|
| Best hyperparameters | `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=4` |
| Training accuracy | 84.7% |
| Test accuracy | 70.0% |
| Overfitting check | Train–test gap > 10% → **model flagged as overfitting** |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Average | 0.69 | 0.80 | 0.74 |
| High | 0.54 | 0.47 | 0.50 |
| Low | 0.84 | 0.67 | 0.74 |

> 📌 **Key finding:** despite grid search and pruning, the tree still shows a meaningful train–test accuracy gap (84.7% vs. 70.0%), correctly flagged by the project's own overfitting check — a useful, honest diagnostic rather than an inflated headline accuracy number. The **High** sales class is the hardest to predict (lowest F1 of 0.50), suggesting it may be underrepresented or harder to separate from **Average**.

## 🗃️ Project Structure

```
.
├── decision_tree_sales_classifier.ipynb   # Entropy/info-gain analysis, GridSearchCV, decision tree
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn gdown
```

> The notebook was developed in Google Colab and fetches the dataset directly via `gdown`; no manual download is required.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn`

## 🔮 Future Improvements

- Address the observed overfitting with stronger regularization (shallower trees, cost-complexity pruning via `ccp_alpha`) or an ensemble method (Random Forest, Gradient Boosting).
- Investigate class imbalance and its effect on the **High** class's lower recall/precision.
- Add feature importance from the fitted tree (`feature_importances_`) alongside the hand-computed information gain for a direct sanity check.
- Try SMOTE or class-weighting to improve minority-class performance.

## 👥 Authors

- [Ghzal Jabbari](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.
