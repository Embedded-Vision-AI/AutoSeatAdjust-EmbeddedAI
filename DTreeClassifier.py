import os, joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#configuration
DATA_XLSX_PATH  = "data/data.xlsx"
MODEL_PATH = "models/pred_model.pth"
HEADER_ROW      = 0

FEATURE_COL   = "Height (cm)"
TARGET_COLS   = [
    "True Long. Distance (cm)",
    "True Seat Angle (from vertical)"
]

#load and prepare data
def load_data(path: str):
    df = pd.read_excel(path, engine="openpyxl", header=HEADER_ROW)
    # clean whitespace around column names
    df.columns = [str(c).strip() for c in df.columns]
    required = [FEATURE_COL] + TARGET_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}\nFound: {df.columns.tolist()}")
    df = df.dropna(subset=required)

    X = df[[FEATURE_COL]].values
    y = df[TARGET_COLS].copy()
    return X, y

#train 1 XGBRegressor per target
def train_models(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {}
    for col in TARGET_COLS:
        print(f"[TRAIN] {col}")
        reg = xgb.XGBRegressor(
            tree_method="hist", #use gpu_hist to use GPU
            n_estimators=200,
            max_depth=6,
            random_state=42,
        )
        reg.fit(X_tr, y_tr[col])
        mse = mean_squared_error(y_val[col], reg.predict(X_val))
        print(f"    MSE = {mse:.4f}")
        models[col] = reg
    return models

#pickling models into one
def save_models(models: dict, model_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(models, model_path)
    print(f"[INFO] Models pickled → {model_path}")

if __name__ == "__main__":
    print("[INFO] Loading data…")
    X, y = load_data(DATA_XLSX_PATH)
    print(f"  Samples: {len(X)}   Feature: {FEATURE_COL}")

    print("[INFO] Training regressors…")
    trained = train_models(X, y)

    save_models(trained, MODEL_PATH)
    print("[DONE]")