# Cluade Code Project Context
This file provides context for Claude Code when working on the project

## Project Overview

**Project Type**: Machine Learning Research - Clustering models

**Project Goal**: Compare multiple types of Clustering models

## Project Context

This is a project where given a data set will use four Clustering models and compare the models using metrics.

## Project Directory Setup
Lab3/
├──CLAUDE.md
├──Clustering.ipynb
├──README.md
├──requirements.txt
└──visualizations/

## ML Preferences

### ML Framework and Tools
- **Primary Framework**: scikit-learn
- **Visualization**: matplotlib and seaborn

### Model Selection Philosophy
- K-Means for a baseline model
- DBSCAN
- K-Medoids
- Agglomerative Clustering
- Gaussian Mixture Models

### Validation Strategy
- **CRITICAL**: Never touch the test set during development
- Use validation set for:
  - Hyperparameter tuning
- Only evaluate on test set once for final results

### Evaluation Metrics
- Compare models with 
    - Silouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Index
    - Inertia(K-Means)
    - training time
    - average prediction time
- Should be orginized in a grid with the models as the rows and the metrics as the columns
- Example Grid

| Model   | Silhouette | Davies-Bouldin | Calinski-Harabasz | n_clusters  | Noise Points| Train Time | Avg. Prediction Time |
|---------|------------|----------------|-------------------|-------------|-------------|------------|----------------------|
| K-Means | 0.92       | 0.85           | 310.23            | 4           | 0           | 0.12s      | .09s                 |
| DBSCAN  | 0.94       | 0.88           | 234.23            | 3           | 42          | 2.45s      | .32s                 |
| ...     | ...        | ...            | ...               | ...         | ...         | ...        | ...                  |

## File Setup
- The project will be written in one jupyter notebook
- The notebook will be written into 6 sections
- The notebook will also have Markdown notes through out the file
- Vizualizations should be stored in the Vizualizations folder

### Section One
- **Will be used for Exploratory data analysis and data-preprocessing**
- Will be split into two subsections

#### Section 1.1
- **Used for Exploratory Data Analysis**
- Statistical summary of all features
- Correlation matrix/heatmap
- Identification of outliers
- Missing value analysis
- Class distribution analysis
- Distribution plots for key numerical features
- Categorical feature analysis (unique values, frequency)
- Pairwise Scatter Plots

#### Section 1.2
- **Used for the Data-preprocessing pipeline**
- Used to handle missing values
- Encode categorical variables (one-hot, label encoding, etc.)
- Feature scaling/normalization
- Address class imbalance (if applicable):
    - SMOTE (Synthetic Minority Over-sampling)
    - Undersampling
    - Class weights
- Two Train/test split with stratification (one with preprocessing and one without preprocessing)

### Section Two
- **Will be used for the baseline K-Means**
- Train multiple models with K values from 2-12
- Training and test performance metrics
- Visualizations
    - Elbow plot(Inertia vs K)
    - Silhouette vs K
    - Cluster Visualizations
    - Silhouette plot for chosen K

### Section Three
- **Will be used for the other four clustering models**
- Train all four other classification models with default hyerparameters and find optimal Parameter Selection
- Record Training time
- Calculate Evaluation Metrics
- For K-Medoids:
    - Use Elbow Method
    - Silhouette analysis
- For DBSCAN:
    - K-distance plot for eps selection
    - Sensitivity analysis for min_samples
- For Hierarchical Clustering:
    - Dendrogram visualization
    - Linkage method comparison (ward, complete, average, single)
    - Cut-off selection for number of clusters
- For Gaussian Mixture Models:
    - BIC/AIC for model selection
    - Compare different covariance types
- Visualizations for each Model
    - 2D scatter plot with cluster labels (use PCA if >2 features)
    - Cluster boundaries (if applicable)
    - Highlight outliers/noise points (for DBSCAN)

### Section Four
- **Will be used for displaying the Evaluation metrics**
- Metrics table
- Cluster Size Comparison across algorithms
- Cluster Stabillity Analysis
    - Bootstrap resampling to assess cluster stability
    - Compare cluster assignments across multiple runs (for non-deterministic algorithms)
- Visualizations
    - Bar charts comparing metrics across algorithms
    - Cluster visualizations for ALL algorithms (use subplots, 2x2 grid)
    - Silhouette plots for ALL algorithms
    - Dendrogram (for hierarchical)
    - Stacked bar chart or pie charts showing cluster proportions
    - Learning Curves: For your top 2 models

### Section Five
- **Will be used for hyperparameter tuning**
- Use the best performing model
- Use internal metrics to optimize the model
- document:
    - Initial hyperparameters vs. optimal hyperparameters
    - Performance improvement (show metrics before/after)
    - Sensitivity analysis: How do results change with parameter variations?
    - Visualization of parameter search results (heatmap if 2 parameters)
    - Computational cost analysis

### Section Six
- **Will be used for Cluster Interpretation and Profiling**
- For the best model, create detailed cluster profiles
- Create a Cluster Profile Table
- Feature Distribution by Cluster
    - Box plots for each feature, grouped by cluster
    - Radar/spider charts comparing cluster centroids
    - Parallel coordinates plot
- Cluster Naming/Labeling
- Statistical Validation
    - ANOVA or Kruskal-Wallis test: Are feature differences between clusters significant?
    - Effect size (eta-squared) for each feature
    - Identify which features best discriminate between clusters

## Data Sources

Mall Customer Segmentation Data from Kaggle

### Getting Data from Data Sources
Use the following code to get the data
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
```

## Git
- **The project will have a git repository**
- Eveytime a update is made to the project you will make a commit and explain the updates
- **Never commit raw data to git**

## Miscellaneous

### README
- The Readme should be updated everytime a change is made to the project
- The Redme will contain all pertinent information

### Requirements.txt
- The requriements.txt will be updated everytime a library is added or removed from the project

**Last Updated**: 2025-12-7

**Note**: This file should be updated whenever project goals, standards, or preferences change.
