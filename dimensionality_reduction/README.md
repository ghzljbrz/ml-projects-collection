# 🧩 Dimensionality Reduction on Fashion-MNIST: PCA, LDA & t-SNE

**A comparative study of dimensionality-reduction techniques on Fashion-MNIST — from a from-scratch PCA implementation and image denoising to LDA-based class separability analysis and t-SNE visualization.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-from--scratch%20PCA-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-PCA%20%7C%20LDA%20%7C%20t--SNE-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-success?style=for-the-badge">
</p>

---

## 📌 Overview

High-dimensional image data (784 pixels per Fashion-MNIST image) is expensive to work with and hard to visualize directly. This project explores **three fundamentally different dimensionality-reduction techniques** — each optimizing for a different goal — and evaluates them on the same dataset:

- **PCA** — unsupervised, variance-maximizing compression (and, as a side effect, image denoising).
- **LDA** — supervised, class-separability-maximizing projection.
- **t-SNE** — non-linear, visualization-focused embedding.

The guiding questions:

> How many components does PCA actually need to preserve most of Fashion-MNIST's information? Can that same PCA compression be used to denoise noisy images? And when the goal shifts from *preserving variance* to *separating classes*, how much better does LDA do — and how does that compare to a purely visual, non-linear technique like t-SNE?

## ✨ What This Project Demonstrates

- 🧮 **PCA from first principles** — implementing Principal Component Analysis via eigendecomposition/SVD in NumPy, computing explained variance ratios, and determining the minimum number of components needed to retain 90% of the dataset's variance.
- 🖼️ **PCA for image denoising** — adding controlled Gaussian noise to sample images, then using a PCA reconstruction (fit on the full dataset) to recover a cleaner version of each noisy image, visually and quantitatively.
- 🎯 **Supervised dimensionality reduction with LDA** — projecting Fashion-MNIST into a low-dimensional space explicitly optimized for class separability, then validating that projection with a downstream Logistic Regression classifier.
- 📐 **Class-separability analysis via scatter matrices** — computing within-class (`Sw`) and between-class (`Sb`) scatter matrices by hand and using `trace(Sw⁻¹Sb)` as a quantitative separability metric across different numbers of LDA components.
- 🗺️ **Non-linear embedding with t-SNE** — visualizing the full 10-class dataset in 2D to qualitatively compare cluster separation against the linear PCA/LDA projections.
- ⚖️ **Head-to-head technique comparison** — contrasting an unsupervised, variance-based method (PCA) against a supervised, separability-based method (LDA) against a non-linear, visualization-only method (t-SNE) on identical data.

## 🗂️ Dataset

- **Source:** Fashion-MNIST (via `sklearn.datasets.fetch_openml`) — 70,000 grayscale 28×28 images of clothing items across 10 categories.
- **Preprocessing:** pixel normalization to `[0, 1]`, flattening images to 784-dimensional vectors, and (for noise/denoising experiments) synthetic Gaussian noise (σ = 0.2) added to sample images.

## 🧪 Methodology

| Stage | Description |
|---|---|
| **1. Noise characterization** | Sample one image per class and add Gaussian noise to study PCA's denoising potential. |
| **2. PCA from scratch** | Implement PCA manually; compute the cumulative explained-variance curve and find the minimum number of components for 90% variance retention. |
| **3. PCA via scikit-learn** | Cross-check the from-scratch results against scikit-learn's `PCA`. |
| **4. PCA denoising** | Fit PCA (100 components) on the full dataset; reconstruct noisy sample images and compare against the originals. |
| **5. LDA for classification** | Project the full dataset with `LinearDiscriminantAnalysis`, then train a Logistic Regression classifier on the LDA-transformed features. |
| **6. Separability matrices** | Compute within-class and between-class scatter matrices (`Sw`, `Sb`) and their `trace(Sw⁻¹Sb)` separability score for both PCA and LDA projections. |
| **7. Optimal LDA dimensionality** | Sweep the number of LDA components and track how the separability score changes. |
| **8. t-SNE visualization** | Embed the full dataset in 2D with t-SNE for a qualitative, non-linear comparison. |

## 🤖 Notebook

### Dimensionality Reduction Study — `dimensionality_reduction_pca_lda_tsne.ipynb`
Covers the complete pipeline above: from-scratch and library PCA, PCA-based image denoising, LDA-based supervised projection and classification, scatter-matrix-based separability analysis, and t-SNE visualization — all on Fashion-MNIST.

## 📈 Results Snapshot

| Analysis | Result |
|---|---|
| Components needed for 90% variance (PCA, from scratch) | **7 components** |
| LDA components used for classification | 9 (n_classes − 1) |
| Explained variance ratio, top LDA component | 44.8% |
| Logistic Regression accuracy on LDA-projected features | **82.9%** |

> 📌 **Key finding:** a Logistic Regression model trained on just the **9 supervised LDA components** achieves **82.9% accuracy** on 10-class Fashion-MNIST classification — a strong result given the massive reduction from 784 raw pixel dimensions, and a clear illustration of why a *supervised* projection built explicitly for class separability can be far more useful for downstream classification than an unsupervised one built purely to preserve variance.

## 🗃️ Project Structure

```
.
├── dimensionality_reduction_pca_lda_tsne.ipynb   # PCA, PCA-denoising, LDA, separability analysis, t-SNE
└── README.md
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone [https://github.com/ghzljbrz/ml-projects-collection].git
cd ml-projects-collection

# Install dependencies
pip install numpy pandas scikit-learn matplotlib
```

> The notebook fetches Fashion-MNIST directly via `sklearn.datasets.fetch_openml`; no manual download is required (the first run may take a few minutes to download and cache the dataset).

## 🛠️ Tech Stack

`Python` · `NumPy` · `scikit-learn` · `Matplotlib`

## 🔮 Future Improvements

- Quantify PCA denoising with a numeric metric (e.g. PSNR or SSIM against the clean originals) rather than visual comparison alone.
- Extend the LDA vs. PCA comparison with additional downstream classifiers (SVM, Random Forest) to check whether the 82.9% accuracy result generalizes.
- Add UMAP as a third non-linear embedding technique alongside t-SNE for a broader visualization comparison.
- Run t-SNE with multiple perplexity values to study its effect on cluster separation.

## 👥 Authors

- [Ghzal Jabbari](https://github.com/ghzljbrz)
- [Asal Sanei](https://github.com/Asal-Sanei)

## 📄 License

This project is available under the MIT License — feel free to explore, fork, and build on it.
