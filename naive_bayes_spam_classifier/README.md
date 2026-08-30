# 📧 Spam Detection with Naive Bayes (From Scratch vs. scikit-learn)

**A text-classification project that detects spam SMS messages, comparing a hand-built Multinomial Naive Bayes classifier against scikit-learn's implementation.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-from--scratch%20Naive%20Bayes-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-MultinomialNB-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="NLTK" src="https://img.shields.io/badge/NLTK-Text%20Processing-154F5B?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

This project tackles a classic NLP problem — **SMS spam detection** — using the **Naive Bayes** algorithm, one of the most widely used probabilistic classifiers for text data. Rather than relying only on a library implementation, the project **derives and implements Multinomial Naive Bayes from first principles in NumPy** (Bayes' theorem, class priors, Laplace-smoothed likelihoods, and log-space posterior computation), then validates it against scikit-learn's battle-tested `MultinomialNB`.

The guiding question:

> Does a from-scratch implementation of Multinomial Naive Bayes, built directly from Bayes' theorem, match the predictive performance of a mature library implementation on a real spam-classification task?

## ✨ What This Project Demonstrates

- 🧮 **Probabilistic ML from first principles** — a custom `MultiNB` class implementing class priors, word-count likelihoods with Laplace (add-one) smoothing, and log-space posterior scoring to avoid numerical underflow — no `sklearn.fit()` shortcuts.
- ✍️ **Text preprocessing & vectorization** — converting raw SMS text into a bag-of-words feature matrix using `CountVectorizer` (stop-word removal, lowercasing).
- ⚖️ **Model validation methodology** — training both the custom and scikit-learn Naive Bayes classifiers on an identical train/test split and feature matrix to enable a fair, apples-to-apples comparison.
- 📊 **Full evaluation suite** — precision, recall, F1-score (via `classification_report`) and confusion matrices for both models, not just raw accuracy.
- 🧹 **Practical data cleaning** — handling a real-world messy CSV (dropping empty/unnamed columns, encoding fixes).

## 🗂️ Dataset

- **Task:** Binary text classification — **ham** (legitimate) vs. **spam** SMS messages.
- **Format:** CSV of labeled SMS messages, loaded with `latin-1` encoding.
- **Split:** 80% train / 20% test (1,115 test messages: 977 ham, 138 spam).

## 🧪 Methodology

| Stage | Description |
|---|---|
| **1. Loading** | Read the raw SMS CSV and drop unused/empty columns. |
| **2. Splitting** | 80/20 train/test split on the raw text and labels. |
| **3. Vectorization** | Fit a `CountVectorizer` (English stop words removed) on the training text; transform both train and test sets into bag-of-words matrices. |
| **4. Custom model** | Implement Multinomial Naive Bayes from scratch: compute class priors, per-feature word counts per class, and predict via Laplace-smoothed log-posteriors. |
| **5. Library model** | Train scikit-learn's `MultinomialNB` on the same feature matrices. |
| **6. Evaluation** | Compare both models via classification reports and confusion matrices. |

## 🤖 Notebook

### Naive Bayes Spam Classifier — `naive_bayes_spam_classifier.ipynb`
Implements and compares two Multinomial Naive Bayes classifiers on the same spam dataset:
- **`MultiNB` (from scratch):** a NumPy class computing `P(class)` priors and `P(word | class)` likelihoods with Laplace smoothing, predicting via `argmax` over log-space posteriors — directly mirroring the Naive Bayes derivation from Bayes' theorem.
- **`MultinomialNB` (scikit-learn):** the reference library implementation, trained on the identical feature matrix for a controlled comparison.

## 📈 Results Snapshot

| Model | Precision (spam) | Recall (spam) | F1 (spam) | Accuracy |
|---|---|---|---|---|
| Custom `MultiNB` (from scratch) | 0.98 | 0.91 | 0.94 | **99%** |
| scikit-learn `MultinomialNB` | 0.98 | 0.91 | 0.94 | **99%** |

> 📌 **Key finding:** the from-scratch implementation achieves **identical performance** to scikit-learn's `MultinomialNB` on this dataset — strong evidence that the underlying Naive Bayes math (priors, Laplace-smoothed likelihoods, log-posterior scoring) was implemented correctly.

## 🗃️ Project Structure

```
.
├── naive_bayes_spam_classifier.ipynb   # Custom vs. scikit-learn Multinomial Naive Bayes for spam detection
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install numpy pandas scikit-learn nltk matplotlib gdown
```

> The notebook was developed in Google Colab and fetches the dataset directly via `gdown`; no manual download is required.

## 🛠️ Tech Stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `NLTK` · `Matplotlib`

## 🔮 Future Improvements

- Extend text preprocessing with stemming/lemmatization and TF-IDF weighting (already imported but not yet applied) to compare against raw bag-of-words counts.
- Add k-fold cross-validation for more robust performance estimates.
- Test robustness on an imbalanced-focused metric (e.g. PR-AUC) given the class imbalance between ham and spam.
- Package the custom `MultiNB` implementation as a small, reusable, `scikit-learn`-compatible estimator.

## 👥 Authors

- [Ghzal Jabbari](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.
