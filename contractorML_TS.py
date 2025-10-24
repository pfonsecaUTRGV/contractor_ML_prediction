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
from sklearn.inspection import permutation_importance
import sklearn

#  Load dataset 
df = pd.read_csv("audits_english_dates.csv")
df = df.fillna('')

# Parse and extract timestamp features 
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce') 
df = df.dropna(subset=['date'])  # drop rows where date parsing failed

# Extract useful temporal features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

print("Added timestamp-based features:", 
      ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'month_sin', 'month_cos'])


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

# --- Numeric time-based features ---
time_features = df[['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'month_sin', 'month_cos']].values

# --- Combine categorical, WBS embeddings, and time features ---
X_cat = cat_features
X_emb = wbs_reduced
X = np.hstack([X_cat, X_emb, time_features])

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
#svr = SVR()
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

'''
param_grid_svr = {
    'kernel': ['rbf', 'linear'],
    'C': [1, 10],
    'epsilon': [0.1, 0.2],
}
'''
param_grid_lgbm = {
    'num_leaves': [31, 50],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

# --- Grid search and train ---

print("\nTuning Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)

print("\nTuning XGBoost...")
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best XGB params:", grid_xgb.best_params_)

'''
print("\nTuning SVR...")
grid_svr = GridSearchCV(svr, param_grid_svr, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_svr.fit(X_train_scaled, y_train)
best_svr = grid_svr.best_estimator_
print("Best SVR params:", grid_svr.best_params_)
'''
print("\nTuning LightGBM...")
grid_lgbm = GridSearchCV(lgbm, param_grid_lgbm, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_lgbm.fit(X_train_scaled, y_train)
best_lgbm = grid_lgbm.best_estimator_
print("Best LightGBM params:", grid_lgbm.best_params_)

# --- Predictions ---
y_pred_rf = best_rf.predict(X_test_scaled)
y_pred_xgb = best_xgb.predict(X_test_scaled)
#y_pred_svr = best_svr.predict(X_test_scaled)
y_pred_lgbm = best_lgbm.predict(X_test_scaled)

# --- Metrics ---
def tolerance_accuracy(y_true, y_pred, tolerance=5):
    return np.mean(np.abs(y_true - y_pred) <= tolerance)

results_df = pd.DataFrame({
    #'Model': ['Random Forest', 'XGBoost', 'SVR', 'LightGBM'],
    'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
    'MAE': [
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_xgb),
        #mean_absolute_error(y_test, y_pred_svr),
        mean_absolute_error(y_test, y_pred_lgbm),
    ],
    'RMSE': [
        sqrt(mean_squared_error(y_test, y_pred_rf)),
        sqrt(mean_squared_error(y_test, y_pred_xgb)),
       # sqrt(mean_squared_error(y_test, y_pred_svr)),
        sqrt(mean_squared_error(y_test, y_pred_lgbm)),
    ],
    'R2': [
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb),
        #r2_score(y_test, y_pred_svr),
        r2_score(y_test, y_pred_lgbm),
    ],
    'Accuracy': [
        tolerance_accuracy(y_test, y_pred_rf, tolerance=5),
        tolerance_accuracy(y_test, y_pred_xgb, tolerance=5),
        #tolerance_accuracy(y_test, y_pred_svr, tolerance=5),
        tolerance_accuracy(y_test, y_pred_lgbm, tolerance=5),
    ]
})

print("\nFinal Comparison Table:")
print(results_df)

# ------------------- PLOTS -------------------

models = {
    "Random Forest": (y_pred_rf, best_rf),
    "XGBoost": (y_pred_xgb, best_xgb),
    #"SVR": (y_pred_svr, best_svr),
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

'''
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

'''

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


# ------------------- FEATURE IMPORTANCE -------------------

'''
# Build feature names
cat_feature_names = encoder.get_feature_names_out(['Kpi', 'contractor'])
pca_feature_names = [f'WBS_PC{i+1}' for i in range(wbs_reduced.shape[1])]
time_feature_names = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'month_sin', 'month_cos']
feature_names = np.concatenate([cat_feature_names, pca_feature_names, time_feature_names])

# --- Random Forest ---
importances_rf = best_rf.feature_importances_
indices_rf = np.argsort(importances_rf)[-20:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices_rf)), importances_rf[indices_rf], align='center')
plt.yticks(range(len(indices_rf)), feature_names[indices_rf])
plt.title('Random Forest - Top 20 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# --- XGBoost ---
importances_xgb = best_xgb.feature_importances_
indices_xgb = np.argsort(importances_xgb)[-20:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], align='center', color='orange')
plt.yticks(range(len(indices_xgb)), feature_names[indices_xgb])
plt.title('XGBoost - Top 20 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# --- LightGBM ---
importances_lgbm = best_lgbm.feature_importances_
indices_lgbm = np.argsort(importances_lgbm)[-20:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices_lgbm)), importances_lgbm[indices_lgbm], align='center', color='green')
plt.yticks(range(len(indices_lgbm)), feature_names[indices_lgbm])
plt.title('LightGBM - Top 20 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

'''

'''
# --- SVR (Permutation Importance) ---
print("\nComputing Permutation Importance for SVR...")
perm_svr = permutation_importance(best_svr, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances_svr = perm_svr.importances_mean
indices_svr = np.argsort(importances_svr)[-20:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices_svr)), importances_svr[indices_svr], align='center', color='purple')
plt.yticks(range(len(indices_svr)), feature_names[indices_svr])
plt.title('SVR (Permutation Importance) - Top 20 Features')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
'''

'''
# Function to get top N features with names and scores
def get_top_features(importances, feature_names, top_n=10):
    indices = np.argsort(importances)[-top_n:][::-1]
    return [(feature_names[i], importances[i]) for i in indices]

# Collect top features for each model
top_rf = get_top_features(importances_rf, feature_names)
top_xgb = get_top_features(importances_xgb, feature_names)
top_lgbm = get_top_features(importances_lgbm, feature_names)
#top_svr = get_top_features(importances_svr, feature_names)

# Build DataFrame
df_summary = pd.DataFrame({
    "RandomForest": [f"{f} ({v:.4f})" for f, v in top_rf],
    "XGBoost": [f"{f} ({v:.4f})" for f, v in top_xgb],
    "LightGBM": [f"{f} ({v:.4f})" for f, v in top_lgbm],
   # "SVR (Permutation)": [f"{f} ({v:.4f})" for f, v in top_svr],
})

print("\n=== Top 10 Feature Importances per Model ===")
print(df_summary.to_string(index=False))


'''
plt.figure(figsize=(10,5))
df.groupby('year')['Grade'].mean().plot(marker='o')
plt.title('Average Contractor Grade Over Time')
plt.ylabel('Average Grade')
plt.xlabel('Year')
plt.grid(True)
plt.show()


# Optionally, export to CSV for paper inclusion
#df_summary.to_csv("feature_importances_summary.csv", index=False)
