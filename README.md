# ml-leakage-pipeline
# Data Leakage Evaluation

## Project Overview
This project demonstrates the impact of **data leakage** in machine learning workflows and how to fix it using a **Pipeline-based approach**. It also evaluates model reliability using **cross-validation** and explores **Decision Tree depth tuning**.

---

##  Objectives
- Identify and reproduce data leakage
- Fix the workflow using Pipeline
- Evaluate model using cross-validation
- Analyze Decision Tree performance

---

## Dataset
Synthetic dataset generated using:
```
python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
```

## Task 1 — Data Leakage (Flawed Approach)

Applied StandardScaler on the entire dataset before splitting
## Problem:
Test data information leaked into training
Results become overly optimistic and unrealistic
## Key Insight:
Never preprocess data before splitting — it leads to data leakage.

## Task 2 — Fixed Workflow Using Pipeline
## Solution:
* Split data using train_test_split
* Used Pipeline to combine:
   - StandardScaler
   - LogisticRegression
* Applied 5-fold cross-validation
  
## Evaluation Metrics:
* Mean Accuracy
* Standard Deviation
## Key Insight:
Pipelines ensure preprocessing happens inside each fold, preventing leakage.

## Task 3 — Decision Tree Depth Experiment
## Tested depths:
max_depth = 1
max_depth = 5
max_depth = 20

## Key Insight:
A moderate depth (e.g., 5) provides the best balance between bias and variance.

```
ml-leakage-pipeline-roshnara/
│
├── leakage_pipeline.py   # Main implementation file
├── README.md             # Project documentation
```
