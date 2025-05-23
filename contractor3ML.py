import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import sklearn

# Load dataset
df = pd.read_csv("audits_english.csv")
df = df.fillna('')

# OneHotEncoder with version compatibility
version = sklearn.__version__
major, minor, *_ = version.split(".")
minor = int(minor)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# BERT embeddings for WBS
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# Apply PCA to reduce embedding dimensionality
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)
print(f"PCA reduced embeddings shape: {wbs_reduced.shape}")

# Combine features
X_cat = cat_features
X_emb = wbs_reduced
X = np.hstack([X_cat, X_emb])
y = df['Grade'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression with scaled features
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression (scaled + PCA) MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}, R2: {r2_lr:.2f}")

# Define models for tuning
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter grids
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

# Grid search for Random Forest
print("\nTuning Random Forest...")
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)

y_pred_rf = best_rf.predict(X_test_scaled)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest (tuned) MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}, R2: {r2_rf:.2f}")

# Grid search for XGBoost
print("\nTuning XGBoost...")
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best XGB params:", grid_xgb.best_params_)

y_pred_xgb = best_xgb.predict(X_test_scaled)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost (tuned) MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}, R2: {r2_xgb:.2f}")

# Optional: plot predictions vs true values for best model
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Grade")
plt.ylabel("Predicted Grade")
plt.title("XGBoost (Tuned): Predicted vs True Grades")
plt.show()
