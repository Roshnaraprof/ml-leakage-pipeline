# ml-leakage-pipeline
Data Leakage Evaluation
Task 1 — Data Leakage (Flawed Approach)
What was done:
Applied StandardScaler on the entire dataset before splitting
Problem:
Test data information leaked into training
Results become overly optimistic and unrealistic
Key Insight:
Never preprocess data before splitting — it leads to data leakage.

Task 2 — Fixed Workflow Using Pipeline
Solution:
Split data using train_test_split
Used Pipeline to combine:
StandardScaler
LogisticRegression
Applied 5-fold cross-validation
Evaluation Metrics:
Mean Accuracy
Standard Deviation
Key Insight:
Pipelines ensure preprocessing happens inside each fold, preventing leakage.

Task 3 — Decision Tree Depth Experiment
Tested depths:
max_depth = 1
max_depth = 5
max_depth = 20

Key Insight:
A moderate depth (e.g., 5) provides the best balance between bias and variance.

'''
ml-leakage-pipeline-roshnara/
│
├── leakage_pipeline.py   # Main implementation file
├── README.md             # Project documentation
'''
