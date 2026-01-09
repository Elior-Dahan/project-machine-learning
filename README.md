# Road Accident Analysis Using Machine Learning

## Project Overview

Road traffic accidents are one of the most critical public safety challenges worldwide, causing loss of life, physical injuries, and significant economic damage. While much of the current technological focus in road safety emphasizes driver behavior and real-time intervention systems, this project deliberately shifts the analytical perspective toward road infrastructure and environmental conditions as deterministic and engineerable risk factors.

The central goal of this project is to model and analyze the complex interaction between pre-crash road characteristics, environmental conditions, and accident severity. Rather than treating machine learning solely as a passive prediction tool, the project demonstrates its potential use as a simulation engine that can support proactive infrastructure planning and risk mitigation.

The project integrates supervised learning models for accident severity classification with unsupervised learning techniques to uncover hidden structural and spatial patterns in historical crash data.

---

## Dataset Description

**Data Source:**  
Maryland State Police – Automated Crash Reporting System (ACRS), accessed via the Socrata Open Data API (Data.gov).

**Dataset Scope:**  
- 117,000 crash records  
- Time period: 2015–2025  
- Initial dimensionality: 37 features  

**Target Variable (Supervised Learning):**  
- `acrs_report_type`  
  - Property Damage Crash  
  - Injury Crash  
  - Fatal Crash  

**Data Storage Strategy:**  
To ensure reproducibility and stability, the raw data was locally frozen and stored in Parquet format, preserving the original schema and enabling efficient downstream processing.

---

## Project Structure

```
PROJECT-MACHINE-LEARNING/
│
├── 01_preprocessing.ipynb
│   Data ingestion, exploratory analysis, deterministic feature engineering, standardization,
│   and data type casting.
│
├── 02_supervised_models.ipynb
│   Training, tuning, and evaluation of supervised classification models.
│
├── 03_unsupervised_models.ipynb
│   Unsupervised learning, including K-Prototypes clustering and DBSCAN spatial analysis.
│
├── montgomery_crash_reporting.parquet
│   Frozen raw dataset.
│
├── montgomery_crash_reporting_processed.parquet
│   Processed dataset used for modeling.
│
├── basic_kproto_model_set1_k7.pkl
├── new_rep_kproto_model_set1_k7.pkl
│   Saved K-Prototypes clustering models.
│
├── requirements.txt
│   Python dependencies required to reproduce the project.
│
└── README.md
```

---

## Methodology

### Data Preprocessing and Feature Engineering

A domain-driven preprocessing approach was adopted to support both prediction and future simulation use cases. The preprocessing emphasizes determinism, interpretability, and prevention of information leakage.

Key steps include:
- Manual deterministic mapping of categorical variables to ensure semantic consistency
- Logical completion of missing values based on cross-column domain knowledge rather than purely statistical imputation
- Explicit separation between pre-crash and post-crash features to prevent data leakage in supervised models
- Temporal feature engineering using cyclic transformations (sine and cosine) for hour, day, and month variables
- Explicit treatment of missing values as informative signals where appropriate

---

## Supervised Learning

### Models Evaluated

The following supervised classification models were evaluated to provide coverage across multiple algorithmic families:
- CatBoost
- LightGBM
- Random Forest
- Logistic Regression
- Categorical Naive Bayes

### Handling Class Imbalance

The dataset exhibits severe class imbalance, with fatal and injury crashes representing a minority of observations. To address this:
- Class weighting was applied to tree-based and linear models
- Random oversampling was used exclusively for the Naive Bayes model due to library limitations

### Evaluation Strategy

Accuracy was intentionally avoided as a primary metric due to its inadequacy under class imbalance. The evaluation focused on:
- **Primary metric:** F1-Macro
- Recall for casualty-related classes
- Average Precision (AP) for hyperparameter optimization

---

## Unsupervised Learning

### K-Prototypes Clustering

K-Prototypes was used to identify accident profiles based on mixed numerical and categorical features.

Key configuration:
- Optimal number of clusters: K = 7
- Initialization method: Cao
- Numerical features normalized to ensure balanced distance calculations
- Spatial coordinates excluded to avoid geographic clustering

### DBSCAN Spatial Analysis

DBSCAN was applied to identify high-risk spatial hotspots.

Configuration highlights:
- Distance metric: Haversine
- Radii defined in meters to preserve physical interpretability
- Multiple parameter settings evaluated for both global and localized clustering

---

## Key Findings

- Distinct accident profiles were identified, including a dedicated cluster of parking-lot crashes characterized by a disproportionately high prevalence of hit-and-run incidents and front-to-front collisions relative to other clusters.
- A separate cluster dominated by 2024–2025 records suggests a change in reporting standards rather than underlying infrastructure behavior.
- Spatial hotspot analysis revealed persistent high-risk locations across a decade, indicating stable infrastructure-related risk factors.
- The combined results demonstrate the feasibility of using machine learning models as simulation engines for infrastructure-focused risk assessment.

---

## How to Run the Project

1. Create and activate a Python virtual environment  
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Execute the notebooks in the following order:
    - `01_preprocessing.ipynb`
    - `02_supervised_models.ipynb`
    - `03_unsupervised_models.ipynb`

---

## Limitations and Future Work
- The models identify statistical associations and do not establish causal relationships  
- Incorporating additional external data sources such as traffic volume, posted speed limits, and documented infrastructure upgrades could improve model robustness and explanatory power  
- Model performance and stability could be further improved by explicitly accounting for changes in crash reporting practices observed around 2024.

---

## Authors

Elior Dahan  
Liraz Shushan  
Industrial Engineering and Management  
2025–2026