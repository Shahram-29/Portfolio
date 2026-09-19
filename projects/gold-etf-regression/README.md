# Gold ETF Price Regression

Applied Statistics & Machine Learning (B9AI102) — CA2. Predicts the gold ETF adjusted close from a 1,718-row, 23-feature dataset.

Models compared: Linear, Ridge and Lasso regression, Support Vector Regression, Decision Tree, Random Forest — each tuned with GridSearchCV and scored on hold-out R² vs 5-fold cross-validated R². The script ends with a model-comparison chart.

**Script:** `gold_etf_regression.py` (pandas, scikit-learn, seaborn).
