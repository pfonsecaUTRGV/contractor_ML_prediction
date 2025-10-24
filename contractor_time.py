import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Define tolerance accuracy function
def tolerance_accuracy(y_true, y_pred, tolerance=5):
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    accuracy = np.mean(within_tolerance)
    return accuracy

# Load dataset
df = pd.read_csv("audits_english_dates.csv")
df = df.fillna('')

#Time series
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# --- Add cyclical encoding for month (better for models) ---
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# Sentence Transformer embeddings for WBS descriptions
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# PCA to reduce dimensionality
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)

# Combine features

time_features = df[['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'month_sin', 'month_cos']].values
X = np.hstack([cat_features, wbs_reduced, time_features])
y = df['Grade'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
svr = SVR()
lgbm = lgb.LGBMRegressor(random_state=42)

# Train Linear Regression
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Train Random Forest with GridSearch
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)

# Train XGBoost with GridSearch
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_scaled)

# Train SVR (default params)
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)

# Train LightGBM (default params)
lgbm.fit(X_train_scaled, y_train)
y_pred_lgbm = lgbm.predict(X_test_scaled)

# Prepare evaluation function to compute all metrics
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    acc = tolerance_accuracy(y_true, y_pred, tolerance=5)
    print(f"{model_name} results:")
    print(f" MAE: {mae:.3f}")
    print(f" RMSE: {rmse:.3f}")
    print(f" R2 Score: {r2:.3f}")
    print(f" Tolerance Accuracy (±5): {acc:.2%}\n")
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Tolerance Accuracy': acc}

# Evaluate all models
results = []
results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression"))
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))
results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost"))
results.append(evaluate_model(y_test, y_pred_svr, "SVR"))
results.append(evaluate_model(y_test, y_pred_lgbm, "LightGBM"))

# Create results dataframe
results_df = pd.DataFrame(results).sort_values(by='MAE').reset_index(drop=True)
print("\nFinal Comparison Table:")
print(results_df)

# Plot MAE, RMSE, R2, and Tolerance Accuracy for models
metrics = ['MAE', 'RMSE', 'R2', 'Tolerance Accuracy']
plt.figure(figsize=(14,10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.barplot(data=results_df, x='Model', y=metric)
    plt.title(metric)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
