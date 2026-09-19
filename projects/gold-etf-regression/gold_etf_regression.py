# =============================================================================
#  CA2 – Applied Statistics and Machine Learning (B9AI102)
#  Gold Price (ETF) Regression Analysis
#  Dataset: Gold Price Prediction (1,718 instances | 23 input features)
#  Target Variable: Adj Close (Gold ETF adjusted closing price)
#  Models: Linear Regression | Support Vector Regression | Decision Tree |
#          Random Forest (bonus) with GridSearchCV hyperparameter tuning
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection   import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing     import StandardScaler, RobustScaler
from sklearn.linear_model      import LinearRegression, Ridge, Lasso
from sklearn.svm               import SVR
from sklearn.tree              import DecisionTreeRegressor
from sklearn.ensemble          import RandomForestRegressor
from sklearn.metrics           import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection        import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  SECTION 1 – DATA PREPARATION")
print("=" * 70)

# 1.1  Load Data
df = pd.read_csv('CA_2_Clean_Dataset.csv')

print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"[INFO] Date range: {df['Date'].iloc[0]}  →  {df['Date'].iloc[-1]}\n")

# 1.2  Drop non-numeric Date column (not useful as a raw feature)
df = df.drop(columns=['Date'])

# 1.3  Inspect data
print("── Column info ──────────────────────────────────────────────────────")
print(df.info())

print("\n── Summary statistics ───────────────────────────────────────────────")
print(df.describe().T.to_string())

# 1.4  Missing value check
print("\n── Missing values ────────────────────────────────────────────────────")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  ✓ No missing values found.")

# 1.5  Outlier detection (IQR method – informational only, no removal)
print("\n── Outlier count per feature (IQR method) ────────────────────────────")
Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
IQR = Q3 - Q1
outlier_counts = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print(outlier_counts.to_string())
print("""
[NOTE] Outliers detected in volume features are genuine market events
       (e.g., high-volatility trading days). Removing them would lose
       economically meaningful signal, so they are retained.
       RobustScaler (used below) minimises their influence on scaling.
""")

# 1.6  Feature / Target split
X = df.drop(columns=['Adj Close'])
y = df['Adj Close']

print(f"Input features (X): {X.shape}   |   Target (y): {y.shape}")
print(f"Feature names: {X.columns.tolist()}\n")

# 1.7  Train / Test split (80 / 20, stratification not needed for regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training set  : {X_train.shape[0]} samples")
print(f"Test set      : {X_test.shape[0]} samples\n")

# 1.8  Feature Scaling
#      RobustScaler chosen because the dataset contains volume columns with
#      extreme outliers; it uses median/IQR instead of mean/std, making it
#      far less sensitive to those outlier values.
scaler = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train → transform train
X_test_sc  = scaler.transform(X_test)        # apply same transform to test

print("[INFO] RobustScaler applied (median/IQR – robust to volume outliers).")
print("[INFO] Scaler fitted on training data only → no data leakage.\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – EXPLORATORY VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  SECTION 2 – EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# 2.1  Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EDA – Gold ETF Adjusted Closing Price (Target Variable)", fontsize=14, fontweight='bold')

axes[0].hist(y, bins=40, color='goldenrod', edgecolor='black', alpha=0.8)
axes[0].set_title("Distribution of Adj Close (Gold ETF)")
axes[0].set_xlabel("Adj Close (USD)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(y.mean(),  color='red',   linestyle='--', label=f"Mean  = {y.mean():.2f}")
axes[0].axvline(y.median(),color='blue',  linestyle='--', label=f"Median= {y.median():.2f}")
axes[0].legend()

axes[1].boxplot(y, vert=True, patch_artist=True,
                boxprops=dict(facecolor='goldenrod', color='black'))
axes[1].set_title("Boxplot of Adj Close (Gold ETF)")
axes[1].set_ylabel("Adj Close (USD)")

plt.tight_layout()
plt.savefig("EDA_Target_Distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] EDA_Target_Distribution.png")

# 2.2  Correlation heatmap
fig, ax = plt.subplots(figsize=(16, 13))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.4, ax=ax, annot_kws={'size': 7})
ax.set_title("Correlation Heatmap – All Features vs Target", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("EDA_Correlation_Heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] EDA_Correlation_Heatmap.png")

# 2.3  Top correlations with target
print("\n── Top 10 features correlated with Adj Close ─────────────────────────")
top_corr = corr['Adj Close'].drop('Adj Close').abs().sort_values(ascending=False)
print(top_corr.head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – HELPER: EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, model, X_tr, y_tr, X_te, y_te, cv_folds=5):
    """Train model, evaluate on test set, compute cross-val R²."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    # Cross-validation on TRAINING set (prevents leakage)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv_folds, scoring='r2')
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    print(f"\n  ─── {name} ──────────────────────────────────────────────────")
    print(f"  R²   (test)        : {r2*100:.2f} %")
    print(f"  MAE  (test)        : {mae:.4f}")
    print(f"  RMSE (test)        : {rmse:.4f}")
    print(f"  CV R² (5-fold)     : {cv_mean*100:.2f} % ± {cv_std*100:.2f} %")

    overfit_gap = r2 - cv_mean
    flag = "⚠ Possible overfit" if overfit_gap > 0.05 else "✓ Stable generalisation"
    print(f"  Test−CV gap        : {overfit_gap*100:.2f} pp  →  {flag}")

    return {
        'Model': name,
        'R2_test': round(r2*100, 2),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'CV_R2_mean': round(cv_mean*100, 2),
        'CV_R2_std':  round(cv_std*100, 2)
    }, y_pred


results_log = []   # collect all result dicts for final comparison table
preds_log   = {}   # collect predictions for actual-vs-predicted plots

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 4 – LINEAR REGRESSION")
print("=" * 70)

# 4.1  Baseline OLS Linear Regression (Method 1)
lr = LinearRegression()
res, pred = evaluate("Linear Regression (OLS)", lr,
                     X_train_sc, y_train, X_test_sc, y_test)
results_log.append(res);  preds_log["Linear Regression"] = pred

# 4.2  Ridge Regression – Method 2 (L2 regularisation via GridSearchCV)
print("""
[RATIONALE] Ridge regression adds an L2 penalty (λ * Σw²) to the OLS cost
function. This shrinks large coefficients, reducing variance / overfitting.
GridSearchCV finds the optimal λ (alpha) via 5-fold cross-validation.
""")

ridge  = Ridge()
alphas = {'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]}
gs_ridge = GridSearchCV(estimator=ridge, param_grid=alphas,
                        scoring='r2', cv=5, n_jobs=-1)
gs_ridge.fit(X_train_sc, y_train)

print(f"  Best α (Ridge) : {gs_ridge.best_params_}")
print(f"  Best CV R²     : {gs_ridge.best_score_*100:.2f} %")

res, pred = evaluate("Ridge Regression (GridSearchCV)", gs_ridge.best_estimator_,
                     X_train_sc, y_train, X_test_sc, y_test)
results_log.append(res);  preds_log["Ridge Regression"] = pred

# 4.3  Lasso Regression – Method 3 (L1 → automatic feature selection)
print("""
[RATIONALE] Lasso (L1 penalty) can shrink coefficients to exactly zero,
performing implicit feature selection — useful when many of the 23 inputs
may be redundant or collinear.
""")

lasso  = Lasso(max_iter=10000)
gs_lasso = GridSearchCV(estimator=lasso, param_grid=alphas,
                        scoring='r2', cv=5, n_jobs=-1)
gs_lasso.fit(X_train_sc, y_train)

print(f"  Best α (Lasso) : {gs_lasso.best_params_}")
print(f"  Best CV R²     : {gs_lasso.best_score_*100:.2f} %")

# Count zero coefficients (features selected to zero)
lasso_coefs = gs_lasso.best_estimator_.coef_
zeroed_out  = (lasso_coefs == 0).sum()
print(f"  Features zeroed by Lasso : {zeroed_out} / {len(lasso_coefs)}")

res, pred = evaluate("Lasso Regression (GridSearchCV)", gs_lasso.best_estimator_,
                     X_train_sc, y_train, X_test_sc, y_test)
results_log.append(res);  preds_log["Lasso Regression"] = pred

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – SUPPORT VECTOR REGRESSION (SVR)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 5 – SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 70)

# 5.1  SVR Method 1 – Manual tuning (mirroring professor's notebook style)
print("""
[RATIONALE] SVR maps inputs to a high-dimensional feature space via a kernel
and fits a tube (ε-insensitive) around the data. Key hyperparameters:
  kernel  – defines the mapping (linear, poly, rbf)
  C       – regularisation: high C = low bias / high variance
  epsilon – tube width: larger ε = fewer support vectors / smoother model
SVR requires scaled features; RobustScaler applied in Section 1.
""")

svr1 = SVR(kernel='rbf', C=10, epsilon=0.1)
svr1.fit(X_train_sc, y_train)
y_pred_svr1 = svr1.predict(X_test_sc)
r2_svr1 = r2_score(y_test, y_pred_svr1)
print(f"\n  SVR Method 1 (rbf, C=10, ε=0.1)")
print(f"  R² (test): {r2_svr1*100:.2f} %")

# 5.2  SVR Method 2 – GridSearchCV
print("\n[METHOD 2] GridSearchCV over kernel, C, epsilon …")

svr_params = {
    'kernel' : ['linear', 'rbf', 'poly'],
    'C'      : [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1]
}
gs_svr = GridSearchCV(estimator=SVR(), param_grid=svr_params,
                      scoring='r2', cv=5, n_jobs=-1, verbose=0)
gs_svr.fit(X_train_sc, y_train)

print(f"  Best params  : {gs_svr.best_params_}")
print(f"  Best CV R²   : {gs_svr.best_score_*100:.2f} %")

res, pred = evaluate("SVR (GridSearchCV – Best)", gs_svr.best_estimator_,
                     X_train_sc, y_train, X_test_sc, y_test)
results_log.append(res);  preds_log["SVR"] = pred

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – DECISION TREE REGRESSOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 6 – DECISION TREE REGRESSOR")
print("=" * 70)

print("""
[RATIONALE] Decision Trees split data recursively on feature thresholds to
minimise MSE (squared_error). Without depth control they overfit perfectly to
training data. max_depth is the primary regularisation hyperparameter.
Note: Decision Trees do NOT require scaling (splits are threshold-based).
""")

# Method 1 – Manual
dt1 = DecisionTreeRegressor(criterion='squared_error', max_depth=10, random_state=42)
res, pred = evaluate("Decision Tree (max_depth=10)", dt1,
                     X_train, y_train, X_test, y_test)   # unscaled – intentional
results_log.append(res);  preds_log["Decision Tree"] = pred

# Method 2 – GridSearchCV
print("\n[METHOD 2] GridSearchCV over max_depth, min_samples_split, min_samples_leaf …")

dt_params = {
    'max_depth'        : [3, 5, 8, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}
gs_dt = GridSearchCV(
    estimator=DecisionTreeRegressor(criterion='squared_error', random_state=42),
    param_grid=dt_params, scoring='r2', cv=5, n_jobs=-1
)
gs_dt.fit(X_train, y_train)

print(f"  Best params  : {gs_dt.best_params_}")
print(f"  Best CV R²   : {gs_dt.best_score_*100:.2f} %")

res, pred = evaluate("Decision Tree (GridSearchCV – Best)", gs_dt.best_estimator_,
                     X_train, y_train, X_test, y_test)
results_log.append(res);  preds_log["Decision Tree (Tuned)"] = pred

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – RANDOM FOREST REGRESSOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 7 – RANDOM FOREST REGRESSOR")
print("=" * 70)

print("""
[RATIONALE] Random Forest is an ensemble of Decision Trees trained on random
bootstrap samples with random feature subsets at each split. This bagging
approach dramatically reduces variance (overfit) vs a single tree, while
maintaining low bias. Key hyperparameters: n_estimators, max_features,
max_depth, min_samples_split.
""")

# Method 1
rf1 = RandomForestRegressor(n_estimators=200, max_features='sqrt',
                             random_state=42, n_jobs=-1)
res, pred = evaluate("Random Forest (n=200, sqrt features)", rf1,
                     X_train, y_train, X_test, y_test)
results_log.append(res);  preds_log["Random Forest"] = pred

# Method 2 – GridSearchCV
print("\n[METHOD 2] GridSearchCV over n_estimators, max_features, max_depth …")

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth'   : [None, 10, 20]
}
gs_rf = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=rf_params, scoring='r2', cv=5, n_jobs=-1
)
gs_rf.fit(X_train, y_train)

print(f"  Best params  : {gs_rf.best_params_}")
print(f"  Best CV R²   : {gs_rf.best_score_*100:.2f} %")

res, pred = evaluate("Random Forest (GridSearchCV – Best)", gs_rf.best_estimator_,
                     X_train, y_train, X_test, y_test)
results_log.append(res);  preds_log["Random Forest (Tuned)"] = pred

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – OVERFITTING AVOIDANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 8 – OVERFITTING AVOIDANCE MECHANISMS")
print("=" * 70)

print("""
Techniques applied in this analysis:
  1. Train/Test Split (80/20)   – holds out unseen data for unbiased evaluation.
  2. k-Fold Cross-Validation    – 5-fold CV on training set; reports mean ± std R².
  3. Regularisation             – Ridge (L2) and Lasso (L1) constrain model weights.
  4. GridSearchCV               – selects hyperparameters by CV score, not test score,
                                  avoiding 'leakage' from test set into tuning.
  5. max_depth / min_samples    – prevent Decision Trees from perfectly memorising data.
  6. Random Forest Bagging      – averages many trees; variance (overfit) collapses.
  7. RobustScaler (SVR/LR)      – prevents large-scale features dominating the model.
""")

# Learning curve for best tree (Decision Tree) to visualise overfit risk
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    gs_dt.best_estimator_, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2', n_jobs=-1
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='royalblue', label='Training R²')
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color='royalblue')
ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='darkorange', label='Validation R²')
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color='darkorange')
ax.set_title("Learning Curve – Best Decision Tree (Overfitting Analysis)", fontsize=13, fontweight='bold')
ax.set_xlabel("Training Set Size")
ax.set_ylabel("R² Score")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Learning_Curve_DecisionTree.png", dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] Learning_Curve_DecisionTree.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 9 – FEATURE IMPORTANCE")
print("=" * 70)

best_rf = gs_rf.best_estimator_
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

print("\n  Top 10 most important features (Random Forest impurity):")
print(importances_sorted.head(10).to_string())

fig, ax = plt.subplots(figsize=(12, 6))
importances_sorted.head(15).plot(kind='bar', ax=ax, color='goldenrod', edgecolor='black')
ax.set_title("Top 15 Feature Importances – Random Forest (Best Model)", fontsize=13, fontweight='bold')
ax.set_ylabel("Mean Decrease in Impurity")
ax.set_xlabel("Feature")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Feature_Importance_RandomForest.png", dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] Feature_Importance_RandomForest.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – PREDICTION WITH NEW DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 10 – PREDICTION WITH NEW (UNSEEN) DATA")
print("=" * 70)

print("""
[APPROACH] A realistic new data instance is created by perturbing the mean
of each feature in the training set (simulating a hypothetical future trading
day with slightly elevated S&P 500, oil futures, and USD index values).
All four best models produce a prediction for this scenario.
""")

# Construct realistic new instance from training-set means + small perturbations
np.random.seed(7)
new_instance_raw = X_train.mean() * (1 + np.random.uniform(-0.02, 0.02, size=X_train.shape[1]))
new_instance_df  = pd.DataFrame([new_instance_raw], columns=X.columns)

print("  New input instance (selected features):")
selected_display = ['SP_Ajclose', 'DJ_Ajclose', 'OF_Price', 'USDI_Price',
                    'GDX_Adj Close', 'USO_Adj Close', 'PLT_Price', 'USB_Price']
print(new_instance_df[selected_display].to_string(index=False))

# Scale for linear / SVR models
new_instance_sc = scaler.transform(new_instance_df)

predictions_new = {
    "Linear Regression (OLS)"          : lr.predict(new_instance_sc)[0],
    "Ridge Regression (Best)"          : gs_ridge.best_estimator_.predict(new_instance_sc)[0],
    "SVR (Best)"                       : gs_svr.best_estimator_.predict(new_instance_sc)[0],
    "Decision Tree (Best)"             : gs_dt.best_estimator_.predict(new_instance_df)[0],
    "Random Forest (Best)"             : gs_rf.best_estimator_.predict(new_instance_df)[0],
}

print("\n  ── Predicted Gold ETF Adj Close for new instance ──────────────────")
for model_name, pred_val in predictions_new.items():
    print(f"  {model_name:<40} → $ {pred_val:.4f} USD")

# Actual mean of y for reference
print(f"\n  [REF] Historical mean Adj Close : $ {y.mean():.4f} USD")
print(f"  [REF] Historical std  Adj Close : $ {y.std():.4f} USD")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – VISUAL COMPARISON: ACTUAL vs PREDICTED
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 11 – ACTUAL vs PREDICTED PLOTS")
print("=" * 70)

plot_models = {
    "Linear Regression": preds_log["Linear Regression"],
    "SVR":               preds_log["SVR"],
    "Decision Tree (Tuned)": preds_log["Decision Tree (Tuned)"],
    "Random Forest (Tuned)": preds_log["Random Forest (Tuned)"],
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Actual vs Predicted – Gold ETF Adj Close", fontsize=15, fontweight='bold')
axes = axes.flatten()

for ax, (name, y_pred) in zip(axes, plot_models.items()):
    r2 = r2_score(y_test, y_pred)
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color='steelblue', edgecolors='none')
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_title(f"{name}\nR² = {r2*100:.2f}%", fontsize=11, fontweight='bold')
    ax.set_xlabel("Actual Adj Close (USD)")
    ax.set_ylabel("Predicted Adj Close (USD)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Actual_vs_Predicted_All_Models.png", dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] Actual_vs_Predicted_All_Models.png")

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Residual Plots – All Models", fontsize=15, fontweight='bold')
axes = axes.flatten()

for ax, (name, y_pred) in zip(axes, plot_models.items()):
    residuals = y_test.values - y_pred
    ax.scatter(y_pred, residuals, alpha=0.4, s=18, color='darkorange', edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f"{name}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Residual_Plots_All_Models.png", dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] Residual_Plots_All_Models.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – FINAL MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 12 – MODEL COMPARISON SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results_log)
results_df = results_df.sort_values('R2_test', ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_r2         = results_df.iloc[0]['R2_test']
best_cv         = results_df.iloc[0]['CV_R2_mean']

print(f"\n  ★ BEST MODEL  : {best_model_name}")
print(f"    Test R²     : {best_r2} %")
print(f"    CV R²       : {best_cv} %")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(14, 7))
x_pos = np.arange(len(results_df))
bars  = ax.bar(x_pos, results_df['R2_test'], color='steelblue',
               edgecolor='black', alpha=0.85, label='Test R²')
ax.errorbar(x_pos, results_df['CV_R2_mean'],
            yerr=results_df['CV_R2_std'],
            fmt='D', color='darkorange', capsize=6, markersize=7,
            label='CV R² (mean ± std)')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Model'], rotation=30, ha='right', fontsize=9)
ax.set_ylabel("R² Score (%)")
ax.set_title("Model Comparison – Test R² vs 5-Fold CV R²", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("Model_Comparison_R2.png", dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] Model_Comparison_R2.png")

print("\n" + "=" * 70)
print("  ALL SECTIONS COMPLETE – CA2 Analysis Finished")
print("=" * 70)
print("""
Files generated:
  EDA_Target_Distribution.png
  EDA_Correlation_Heatmap.png
  Learning_Curve_DecisionTree.png
  Feature_Importance_RandomForest.png
  Actual_vs_Predicted_All_Models.png
  Residual_Plots_All_Models.png
  Model_Comparison_R2.png
""")
