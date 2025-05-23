import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import sklearn
from sklearn.preprocessing import OneHotEncoder



# Load dataset - update filename if needed
df = pd.read_csv("audits_english.csv")

print("=== Dataset sample ===")
print(df.head())
print("\nColumns:", df.columns.tolist())

# Fill missing values just in case
df = df.fillna('')

# Check column names carefully, adjust if needed
print("\nColumn names - exact case matters:")
print(df.columns)

# Encode categorical columns - check names and cases
try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])
except KeyError as e:
    print(f"ERROR: Column not found - {e}")
    raise

print("\nCategorical features shape:", cat_features.shape)
print("Sample categorical feature row:", cat_features[0])

# Create BERT embeddings for 'wbs' column
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_texts = df['wbs'].tolist()
print("\nEncoding WBS texts with BERT...")
wbs_embeddings = model.encode(wbs_texts, show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

print("WBS embeddings shape:", wbs_embeddings.shape)
print("Sample WBS embedding vector (first 5 values):", wbs_embeddings[0][:5])

# Check for NaNs or infinite values in embeddings
print("Any NaNs in embeddings?", np.any(np.isnan(wbs_embeddings)))
print("Any infinite values in embeddings?", np.any(np.isinf(wbs_embeddings)))

# Combine features
X = np.hstack([cat_features, wbs_embeddings])
print("\nCombined feature matrix shape:", X.shape)

# Target variable
y = df['Grade'].values
print("Target y shape:", y.shape)
print("Sample target values:", y[:5])
print("Any NaNs or infs in target?", np.any(np.isnan(y)) or np.any(np.isinf(y)))

# Quick train-test split for debug (using first 100 for train to speed up)
X_train, y_train = X[:100], y[:100]
X_test, y_test = X[100:110], y[100:110]

# Train Linear Regression
print("\nTraining Linear Regression on small sample for debug...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("Sample predictions:", y_pred)
print("True values:", y_test)

# Evaluate model on this small test
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation on small test set:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")
