# Clustering Analysis - Mall Customer Segmentation

A comprehensive machine learning project comparing multiple clustering algorithms on the Mall Customer Segmentation dataset from Kaggle.

## Project Overview

This project explores and compares five different clustering algorithms to segment mall customers based on their characteristics. The goal is to identify distinct customer groups and evaluate which clustering approach performs best for this dataset.

## Clustering Models

- **K-Means** (baseline model)
- **DBSCAN**
- **K-Medoids**
- **Agglomerative Clustering**
- **Gaussian Mixture Models (GMM)**

## Dataset

**Source:** [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) from Kaggle

The dataset contains customer information from a mall including demographics and spending behavior.

## Project Structure

```
Lab4/
├── CLAUDE.md                 # Project context for Claude Code
├── Clustering.ipynb          # Main analysis notebook
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── download_data.py          # Script to download dataset
└── visualizations/           # Generated visualizations
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Lab4
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python download_data.py
```

## Usage

Open and run the Jupyter notebook:
```bash
jupyter notebook Clustering.ipynb
```

## Notebook Structure

### Section 1: Exploratory Data Analysis and Data Preprocessing
- **Section 1.1:** Exploratory Data Analysis ✓
  - Statistical summaries
  - Correlation analysis
  - Outlier identification
  - Missing value analysis
  - Distribution visualizations
  - Pairwise scatter plots

- **Section 1.2:** Data Preprocessing ✓
  - Feature selection (Annual Income & Spending Score)
  - Missing value handling
  - Feature scaling with StandardScaler
  - Two datasets created: original and scaled
  - Preprocessing visualizations and comparisons

### Section 2: Baseline K-Means Model ✓
- Training with K values from 2 to 12
- Elbow Method visualization (Inertia vs K)
- Silhouette Score analysis
- Optimal K selection based on metrics
- Cluster visualizations (original and scaled data)
- Detailed silhouette plot for optimal K
- Comprehensive performance metrics and evaluation

### Section 3: Additional Clustering Models ✓
- **K-Medoids**: PAM algorithm with optimal K selection via silhouette analysis
- **DBSCAN**: K-distance plot and parameter grid search (eps, min_samples)
- **Agglomerative Clustering**: Dendrogram visualization and linkage method comparison
- **Gaussian Mixture Models**: BIC/AIC-based model selection with covariance type optimization
- Comprehensive metrics table comparing all 5 models
- 2x3 grid visualization showing all clustering results

### Section 4: Model Comparison ✓
- Comprehensive metrics comparison table (all 5 models)
- Bar chart visualizations for all metrics
- Cluster size comparison across algorithms (pie charts)
- Cluster stability analysis using bootstrap resampling (30 iterations)
- Side-by-side visualization grid (2x3) showing all clustering results
- Silhouette plots comparison for all models (2x2 grid)
- Learning curves for top 2 models (K-Medoids and K-Means)
- Detailed recommendations and key findings

### Section 5: Hyperparameter Tuning ✓
- Initial hyperparameters documentation for K-Medoids
- Comprehensive grid search (54 parameter combinations tested)
- Parameter space: n_clusters (3-8), init methods (3 types), max_iter (3 values)
- Grid search visualizations (heatmaps, line plots, performance metrics)
- Before vs after performance comparison with improvement metrics
- Parameter sensitivity analysis (n_clusters, init method, max_iter)
- Computational cost analysis and efficiency metrics
- Optimal hyperparameters identified with detailed justification

### Section 6: Cluster Interpretation and Profiling ✓
- Detailed cluster profile tables with comprehensive statistics
- Feature distribution visualizations (box plots for income and spending)
- Radar charts comparing cluster centroids
- Parallel coordinates plots showing customer segment patterns
- Scatter plots with cluster centroids and annotations
- Meaningful cluster naming and labeling based on characteristics
- Statistical validation using ANOVA and effect size analysis
- Feature discrimination analysis identifying most important features
- Comprehensive business insights and marketing recommendations per segment
- Implementation roadmap and expected business outcomes
- Complete project summary and final recommendations

## Evaluation Metrics

Models are compared using:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Inertia (for K-Means)
- Training time
- Average prediction time

## Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **kagglehub** - Dataset download

## Progress

- [x] Project setup
- [x] Dataset download
- [x] Section 1.1: Exploratory Data Analysis
- [x] Section 1.2: Data Preprocessing
- [x] Section 2: Baseline K-Means
- [x] Section 3: Additional clustering models
- [x] Section 4: Model comparison
- [x] Section 5: Hyperparameter tuning
- [x] Section 6: Cluster interpretation

## License

This project is for educational purposes.

## Author

Created as part of a machine learning research project.

---

**Last Updated:** 2025-12-07
