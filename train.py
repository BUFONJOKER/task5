import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import json
from datetime import datetime
import os

if os.path.exists("model/model_scaler_mae.pkl"):
    model = joblib.load("model/model_scaler_mae.pkl")
    version = model['model_version'] + 1
else:
    version = 1

# Load dataset
# Ensure you have pyarrow installed: pip install pyarrow
df = pd.read_csv("data/dataset.csv")

X = df.drop("annual_medical_cost", axis=1)
y = df["annual_medical_cost"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions & Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae}")

# ==========================================
# 2. SAVE ARTIFACTS (Only if training succeeds)
# ==========================================

model_package = {
    "model": model,
    "mae": mae,
    "scaler": scaler,
    'model_version': version,
    'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# save model + scaler + mae
joblib.dump(model_package, "model/model_scaler_mae.pkl")
