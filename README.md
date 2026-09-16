# Telco Customer Churn Prediction

CA1 – Applied Data Analytics.

This project predicts whether a telecom customer will churn using the Telco Customer dataset. The model is a regularised logistic regression (`SGDClassifier` with `log_loss`), trained with SMOTE to handle class imbalance and tuned with grid search.

## Dataset

- File: `Telco_Customer.csv`
- Original size: 7,043 customers, 22 columns
- After cleaning: 7,032 rows (rows with missing `TotalCharges` removed)
- Target: `Churn` (`Yes` / `No`)

Features include demographics, tenure, services, contract type, billing, and charges.

## What the notebook does

1. Load and inspect the data
2. Drop `customerID`
3. Convert `TotalCharges` to numeric and drop missing values
4. Map binary categories to 0/1
5. One-hot encode: `TenureBand`, `MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`
6. Scale features with `StandardScaler`
7. Train/test split (80/20, stratified)
8. Pipeline: **SMOTE → SGDClassifier (elastic-net logistic regression)**
9. `GridSearchCV` with **recall** as the scoring metric
10. Evaluate on the hold-out test set
11. Explore monthly charges vs churn with a boxplot

## Best hyperparameters

| Parameter | Value |
|---|---|
| `alpha` | 0.1 |
| `eta0` | 0.001 |
| `l1_ratio` | 1 |
| `max_iter` | 100 |

Cross-validated recall (best model): **0.871**

## Test-set results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No churn (0) | 0.95 | 0.57 | 0.71 |
| Churn (1) | 0.43 | 0.91 | 0.59 |

- Accuracy: 0.66
- The model was tuned to catch churners (high recall on class 1), so it flags more customers as at-risk and has more false positives.

## Exploratory finding

Median monthly charges:

- Stayed: **64.43**
- Churned: **79.65**

Customers who left tended to pay more per month.
