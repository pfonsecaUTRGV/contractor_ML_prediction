import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# --- Load dataset ---
df = pd.read_csv("audits_english.csv")
df = df.fillna('')

# --- Encode categorical features ---
version = sklearn.__version__
major, minor, *_ = version.split(".")
minor = int(minor)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# --- BERT embeddings for WBS ---
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# --- PCA ---
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)
print(f"PCA reduced embeddings shape: {wbs_reduced.shape}")

# --- Combine features ---
X_cat = cat_features
X_emb = wbs_reduced
X = np.hstack([X_cat, X_emb])
y = df['Grade'].values

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Feature scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Models ---
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
svr = SVR()
lgbm = lgb.LGBMRegressor(random_state=42)

# --- Hyperparameter grids ---
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
}

param_grid_svr = {
    'kernel': ['rbf', 'linear'],
    'C': [1, 10],
    'epsilon': [0.1, 0.2],
}

param_grid_lgbm = {
    'num_leaves': [31, 50],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

# --- Grid search and train ---

print("\nTuning Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)

print("\nTuning XGBoost...")
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best XGB params:", grid_xgb.best_params_)

print("\nTuning SVR...")
grid_svr = GridSearchCV(svr, param_grid_svr, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_svr.fit(X_train_scaled, y_train)
best_svr = grid_svr.best_estimator_
print("Best SVR params:", grid_svr.best_params_)

print("\nTuning LightGBM...")
grid_lgbm = GridSearchCV(lgbm, param_grid_lgbm, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_lgbm.fit(X_train_scaled, y_train)
best_lgbm = grid_lgbm.best_estimator_
print("Best LightGBM params:", grid_lgbm.best_params_)

# --- Predictions ---
y_pred_rf = best_rf.predict(X_test_scaled)
y_pred_xgb = best_xgb.predict(X_test_scaled)
y_pred_svr = best_svr.predict(X_test_scaled)
y_pred_lgbm = best_lgbm.predict(X_test_scaled)

# --- Metrics ---
def tolerance_accuracy(y_true, y_pred, tolerance=5):
    return np.mean(np.abs(y_true - y_pred) <= tolerance)

results_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'SVR', 'LightGBM'],
    'MAE': [
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_xgb),
        mean_absolute_error(y_test, y_pred_svr),
        mean_absolute_error(y_test, y_pred_lgbm),
    ],
    'RMSE': [
        sqrt(mean_squared_error(y_test, y_pred_rf)),
        sqrt(mean_squared_error(y_test, y_pred_xgb)),
        sqrt(mean_squared_error(y_test, y_pred_svr)),
        sqrt(mean_squared_error(y_test, y_pred_lgbm)),
    ],
    'R2': [
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb),
        r2_score(y_test, y_pred_svr),
        r2_score(y_test, y_pred_lgbm),
    ],
    'Accuracy': [
        tolerance_accuracy(y_test, y_pred_rf, tolerance=5),
        tolerance_accuracy(y_test, y_pred_xgb, tolerance=5),
        tolerance_accuracy(y_test, y_pred_svr, tolerance=5),
        tolerance_accuracy(y_test, y_pred_lgbm, tolerance=5),
    ]
})

print("\nFinal Comparison Table:")
print(results_df)

# ------------------- PLOTS -------------------

models = {
    "Random Forest": (y_pred_rf, best_rf),
    "XGBoost": (y_pred_xgb, best_xgb),
    "SVR": (y_pred_svr, best_svr),
    "LightGBM": (y_pred_lgbm, best_lgbm)
}

# 1. Bar plots for MAE, RMSE, R2, Accuracy
metrics_to_plot = ['MAE', 'RMSE', 'R2', 'Accuracy']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i // 2, i % 2]
    sns.barplot(data=results_df, x='Model', y=metric, palette='viridis', ax=ax)
    ax.set_title(f'{metric} by Model')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

# 2. Residual plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, (model_name, (y_pred, _)) in enumerate(models.items()):
    ax = axes[i // 2, i % 2]
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Grade')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residual Plot: {model_name}')
plt.tight_layout()
plt.show()

# 3. Predicted vs Actual scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, (model_name, (y_pred, _)) in enumerate(models.items()):
    ax = axes[i // 2, i % 2]
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('True Grade')
    ax.set_ylabel('Predicted Grade')
    ax.set_title(f'Predicted vs True: {model_name}')
plt.tight_layout()
plt.show()

# 4. Boxplots of absolute errors
abs_errors = {model_name: np.abs(y_test - y_pred) for model_name, (y_pred, _) in models.items()}
abs_errors_df = pd.DataFrame(abs_errors)
plt.figure(figsize=(10, 6))
sns.boxplot(data=abs_errors_df, palette='viridis')
plt.title('Boxplot of Absolute Errors by Model')
plt.ylabel('Absolute Error')
plt.xticks(rotation=45)
plt.show()

# 5. Feature importance (only for RF and XGBoost)
plt.figure(figsize=(12, 6))
importances_rf = best_rf.feature_importances_
indices_rf = np.argsort(importances_rf)[-20:]
plt.barh(range(len(indices_rf)), importances_rf[indices_rf], align='center')
plt.yticks(range(len(indices_rf)), [f'Feat {i}' for i in indices_rf])
plt.title('Random Forest Top 20 Feature Importances')
plt.xlabel('Importance')
plt.show()

plt.figure(figsize=(12, 6))
importances_xgb = best_xgb.feature_importances_
indices_xgb = np.argsort(importances_xgb)[-20:]
plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], align='center', color='orange')
plt.yticks(range(len(indices_xgb)), [f'Feat {i}' for i in indices_xgb])
plt.title('XGBoost Top 20 Feature Importances')
plt.xlabel('Importance')
plt.show()

# 6. Tolerance accuracy vs tolerance threshold curve
tolerance_values = np.arange(0, 11)
plt.figure(figsize=(10, 6))
for model_name, (y_pred, _) in models.items():
    accuracies = [tolerance_accuracy(y_test, y_pred, tol) for tol in tolerance_values]
    plt.plot(tolerance_values, accuracies, label=model_name)

plt.xlabel('Tolerance Threshold')
plt.ylabel('Accuracy')
plt.title('Tolerance Accuracy vs Tolerance Threshold')
plt.legend()
plt.grid(True)
plt.show()





# Prepare individual metric DataFrames
mae_df = results_df[['Model', 'MAE']]
mse_df = results_df[['Model']].copy()
mse_df['MSE'] = [
    mean_squared_error(y_test, y_pred_rf),
    mean_squared_error(y_test, y_pred_xgb),
    mean_squared_error(y_test, y_pred_svr),
    mean_squared_error(y_test, y_pred_lgbm)
]
rmse_df = results_df[['Model', 'RMSE']]
r2_df = results_df[['Model', 'R2']]
accuracy_df = results_df[['Model', 'Accuracy']]

# Save each metric to its own CSV file
mae_df.to_csv("mae_scores.csv", index=False)
mse_df.to_csv("mse_scores.csv", index=False)
rmse_df.to_csv("rmse_scores.csv", index=False)
r2_df.to_csv("r2_scores.csv", index=False)
accuracy_df.to_csv("tolerance_accuracy_scores.csv", index=False)

# --- Export data for Predicted vs Actual and Residual plots ---
predicted_data = {
    "RandomForest": pd.DataFrame({
        "TrueGrade": y_test,
        "PredictedGrade": y_pred_rf,
        "Residual": y_test - y_pred_rf
    }),
    "XGBoost": pd.DataFrame({
        "TrueGrade": y_test,
        "PredictedGrade": y_pred_xgb,
        "Residual": y_test - y_pred_xgb
    }),
    "SVR": pd.DataFrame({
        "TrueGrade": y_test,
        "PredictedGrade": y_pred_svr,
        "Residual": y_test - y_pred_svr
    }),
    "LightGBM": pd.DataFrame({
        "TrueGrade": y_test,
        "PredictedGrade": y_pred_lgbm,
        "Residual": y_test - y_pred_lgbm
    }),
}

# Save each model's data to a separate CSV
for model_name, df in predicted_data.items():
    filename = f"predicted_vs_actual_{model_name.lower()}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")


print("Saved all metrics as individual CSV files.")
