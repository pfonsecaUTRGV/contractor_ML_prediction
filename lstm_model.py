import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt

# --------------------
# LOAD DATA
# --------------------
df = pd.read_csv("audits_english_dates.csv")
df = df.fillna('')

# --- Parse date ---
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])
df = df.sort_values(by=['contractor', 'date'])


# TIME FEATURES

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


# ENCODING

# --- OneHotEncode contractor + KPI ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# --- BERT embeddings for WBS ---
model_bert = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model_bert.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# Reduce dimensionality
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)


# FEATURE MATRIX

time_features = df[['year','month','day','dayofweek','quarter',
                    'is_weekend','month_sin','month_cos']].values

X_full = np.hstack([cat_features, wbs_reduced, time_features])
y_full = df['Grade'].values
contractor_ids = df['contractor'].values

# LSTM SEQUENCES

sequences = []
targets = []
contractors = df['contractor'].unique()

for c in contractors:
    df_c = df[df['contractor'] == c]
    X_c = X_full[df['contractor'] == c]
    y_c = y_full[df['contractor'] == c]

    sequences.append(X_c)
    targets.append(y_c)


max_len = max(len(seq) for seq in sequences)

X_padded = pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='pre')
y_padded = pad_sequences(targets, maxlen=max_len, dtype='float32', padding='pre')


# TRAIN AND TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_padded, test_size=0.2, random_state=42
)

# Only predict last value in each sequence
y_train_last = y_train[:, -1]
y_test_last  = y_test[:, -1]


# BUILD MODEL

n_features = X_train.shape[2]

model = Sequential([
    Masking(mask_value=0.0, input_shape=(max_len, n_features)),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mae']
)

model.summary()

#TRAIN MODEL

history = model.fit(
    X_train, y_train_last,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    verbose=1
)

# EVALUATE

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test_last, y_pred)
rmse = sqrt(np.mean((y_pred - y_test_last)**2))
r2 = r2_score(y_test_last, y_pred)

print("\nLSTM Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

