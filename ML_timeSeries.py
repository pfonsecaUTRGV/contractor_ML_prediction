import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================
# 1. LOAD AND PREPARE DATA
# ======================================
df = pd.read_csv("audits_english_dates.csv")
df = df.fillna('')

# --- Convert and sort by date ---
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])
df = df.sort_values('date').reset_index(drop=True)

# --- Add lag and rolling features for time awareness ---
for lag in [1, 2, 3]:
    df[f'Grade_lag_{lag}'] = df['Grade'].shift(lag)

df['Grade_roll_mean_3'] = df['Grade'].shift(1).rolling(window=3).mean()
df['Grade_roll_std_3']  = df['Grade'].shift(1).rolling(window=3).std()

# Drop first few rows with NaN lag values
df = df.dropna(subset=['Grade_lag_3']).reset_index(drop=True)

# --- Extract timestamp features ---
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print("Time-based, lag, and rolling features added.")

# ======================================
# 2. FEATURE ENGINEERING
# ======================================

# --- Encode categorical features ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# --- BERT embeddings for WBS ---
print("Generating BERT embeddings for WBS...")
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# --- PCA ---
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)
print(f"PCA reduced embeddings shape: {wbs_reduced.shape}")

# --- Combine features ---
time_features = df[['year','month','day','dayofweek','quarter','is_weekend','month_sin','month_cos',
                    'Grade_lag_1','Grade_lag_2','Grade_lag_3','Grade_roll_mean_3','Grade_roll_std_3']].values

X = np.hstack([cat_features, wbs_reduced, time_features])
y = df['Grade'].values

# ======================================
# 3. CHRONOLOGICAL SPLIT
# ======================================
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data ready for time-series modeling.")

# ======================================
# 4. MODEL TRAINING (TIME SERIES CV)
# ======================================
tscv = TimeSeriesSplit(n_splits=5)

rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
svr = SVR()
lgbm = lgb.LGBMRegressor(random_state=42)

param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
param_grid_svr = {'kernel': ['rbf', 'linear'], 'C': [1, 10], 'epsilon': [0.1, 0.2]}
param_grid_lgbm = {'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}

models = {
    "Random Forest": (rf, param_grid_rf),
    "XGBoost": (xgb, param_grid_xgb),
    "SVR": (svr, param_grid_svr),
    "LightGBM": (lgbm, param_grid_lgbm)
}

results = []

for name, (model, params) in models.items():
    print(f"\nTraining {name} with TimeSeriesSplit...")
    grid = GridSearchCV(model, params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "Accuracy": np.mean(np.abs(y_test - y_pred) <= 5)
    })

results_df = pd.DataFrame(results)
print("\nFinal Time-Series Model Comparison:")
print(results_df)

# ======================================
# 5. VISUALIZATIONS
# ======================================
plt.figure(figsize=(10,5))
sns.barplot(data=results_df, x="Model", y="R2", palette="viridis")
plt.title("R² Comparison of Time-Series Models")
plt.show()
