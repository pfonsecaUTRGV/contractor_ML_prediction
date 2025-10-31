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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')


# ===============================
# LOAD AND PREPARE DATA
# ===============================
df = pd.read_csv("audits_english_dates.csv")
df = df.fillna('')

# --- Convert and extract time features ---
# Replace 'Date' with your actual date column name if different
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])
df = df.sort_values(by='date')  # maintain chronological order

# Extract timestamp features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Add cyclic encodings (capture periodicity)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

print(" Timestamp features added.")

# ===============================
# FEATURE ENGINEERING
# ===============================

# --- Encode categorical features ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# --- Generate BERT embeddings for WBS ---
print("Generating BERT embeddings for WBS...")
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# --- PCA dimension reduction ---
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)
print(f"PCA reduced embeddings shape: {wbs_reduced.shape}")

# --- Combine all features ---
time_features = df[['year', 'month', 'day', 'dayofweek', 'quarter',
                    'is_weekend', 'month_sin', 'month_cos',
                    'dow_sin', 'dow_cos']].values

X = np.hstack([cat_features, wbs_reduced, time_features])
y = df['Grade'].values

# ===============================
# TRAIN / TEST SPLIT
# ===============================

# Time-ordered split (not random)
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# --- Feature scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Data prepared for training.")

# ===============================
# MODEL TRAINING AND TUNING
# ===============================

rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
svr = SVR()
lgbm = lgb.LGBMRegressor(
    random_state=42,
    min_gain_to_split=0.0,   # prevents -inf warnings
    min_data_in_leaf=10,     # avoid overfitting small groups
    verbose=-1,              # silence lightgbm internal warnings
    force_row_wise=True      # stabilizes training
)

# --- Hyperparameter grids ---
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1]}
param_grid_svr = {'kernel': ['rbf', 'linear'], 'C': [1, 10], 'epsilon': [0.1, 0.2]}
param_grid_lgbm = {'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}

# --- Grid search and fit ---
print("\nTuning Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)

print("\nTuning XGBoost...")
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best XGB params:", grid_xgb.best_params_)

print("\nTuning SVR...")
grid_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_svr.fit(X_train_scaled, y_train)
best_svr = grid_svr.best_estimator_
print("Best SVR params:", grid_svr.best_params_)

print("\nTuning LightGBM...")
grid_lgbm = GridSearchCV(lgbm, param_grid_lgbm, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_lgbm.fit(X_train_scaled, y_train)
best_lgbm = grid_lgbm.best_estimator_
print("Best LightGBM params:", grid_lgbm.best_params_)

# ===============================
# EVALUATION
# ===============================

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
        tolerance_accuracy(y_test, y_pred_rf),
        tolerance_accuracy(y_test, y_pred_xgb),
        tolerance_accuracy(y_test, y_pred_svr),
        tolerance_accuracy(y_test, y_pred_lgbm),
    ]
})

print("\n Final Comparison Table:")
print(results_df)

# ===============================
# VISUALIZATIONS
# ===============================

models = {
    "Random Forest": (y_pred_rf, best_rf),
    "XGBoost": (y_pred_xgb, best_xgb),
    "SVR": (y_pred_svr, best_svr),
    "LightGBM": (y_pred_lgbm, best_lgbm)
}

metrics_to_plot = ['MAE', 'RMSE', 'R2', 'Accuracy']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i // 2, i % 2]
    sns.barplot(data=results_df, x='Model', y=metric, palette='viridis', ax=ax)
    ax.set_title(f'{metric} by Model')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()
