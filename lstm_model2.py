import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# LOAD DATA
df = pd.read_csv("audits_english_dates.csv")
df = df.fillna('')

# Parse dates and sort by contractor + date
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])
df = df.sort_values(by=['contractor', 'date']).reset_index(drop=True)

#TIME SERIES

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# CYCLIC ENCODINGS
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

print("Time features added.")

#ENCODERS AND EMBEDING

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

print("Generating BERT embeddings for WBS...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = st_model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# PCA
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)
print("PCA reduced embeddings shape:", wbs_reduced.shape)

# TIME FEATURES
time_features = df[['year', 'month', 'day', 'dayofweek', 'quarter',
                    'is_weekend', 'month_sin', 'month_cos',
                    'dow_sin', 'dow_cos']].values

# STACK DATA
X_all = np.hstack([cat_features, wbs_reduced, time_features])
y_all = df['Grade'].values
dates_all = df['date'].values
contractors_all = df['contractor'].values

n_samples, n_features = X_all.shape
print("Total rows:", n_samples, "Features per row:", n_features)

# BUID SEQUENCES
seq_len = 5  # number of past audits to use

X_seqs = []
y_seqs = []
target_dates = []

unique_contractors = df['contractor'].unique()

for contractor in unique_contractors:
    mask = (contractors_all == contractor)
    X_c = X_all[mask]
    y_c = y_all[mask]
    dates_c = dates_all[mask]

    # need at least seq_len + 1 audits to create 1 sample
    if len(X_c) <= seq_len:
        continue

    # sliding window: use seq_len past rows to predict the next one
    for t in range(seq_len, len(X_c)):
        X_seqs.append(X_c[t-seq_len:t])
        y_seqs.append(y_c[t])
        target_dates.append(dates_c[t])

X_seqs = np.array(X_seqs)          # shape: (N_samples, seq_len, n_features)
y_seqs = np.array(y_seqs)          # shape: (N_samples,)
target_dates = np.array(target_dates)

print("Built LSTM dataset: sequences =", X_seqs.shape, "targets =", y_seqs.shape)

# TRAIN AND TEST SPLIT
order = np.argsort(target_dates)
X_seqs = X_seqs[order]
y_seqs = y_seqs[order]
target_dates = target_dates[order]

split_idx = int(0.8 * len(X_seqs))  # 80% oldest for training
X_train, X_test = X_seqs[:split_idx], X_seqs[split_idx:]
y_train, y_test = y_seqs[:split_idx], y_seqs[split_idx:]

print("Train sequences:", X_train.shape, "Test sequences:", X_test.shape)

# SCALING FEATURES
n_train, seq_len, n_feat = X_train.shape
n_test = X_test.shape[0]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_2d = X_train.reshape(-1, n_feat)
X_test_2d = X_test.reshape(-1, n_feat)

X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, seq_len, n_feat)
X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, seq_len, n_feat)

# TRAIN LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, n_feat)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # predicting a single grade

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# EVALUATION
y_pred = model.predict(X_test_scaled).ravel()

mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
tolerance = 5
accuracy = np.mean(np.abs(y_test - y_pred) <= tolerance)

print("\nLSTM Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print(f"Accuracy (±{tolerance} points):", accuracy)

# PLOT
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('LSTM Training and Validation MAE')
plt.legend()
plt.tight_layout()
plt.show()
