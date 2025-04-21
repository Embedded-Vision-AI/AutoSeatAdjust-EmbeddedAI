import os
import time
import numpy as np
import xgboost as xgb
import joblib
import pandas as pd
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# configuration
DATA_XLSX_PATH = "data/data.xlsx"
MODEL_PATH     = "models/pred_model.pth"
FEATURE_COL    = "Height (cm)"
TARGET_COLS    = [
    "True Long. Distance (cm)",
    "True Seat Angle (from vertical)"
]
HEADER_ROW     = 0
CV_FOLDS       = 5

def load_data(path: str):
    df = pd.read_excel(path, engine="openpyxl", header=HEADER_ROW)
    df.columns = [str(c).strip() for c in df.columns]
    required = [FEATURE_COL] + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}\nFound: {df.columns.tolist()}")
    df = df.dropna(subset=required)
    X = df[[FEATURE_COL]].values
    y = df[TARGET_COLS].copy()
    return X, y

def train_models(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {}

    for col in TARGET_COLS:
        print(f"[TRAIN] {col}")

        # measure training time
        start_time = time.time()

        reg = xgb.XGBRegressor(
            tree_method="hist",  # switch to gpu_hist if you have GPU
            n_estimators=200,
            max_depth=6,
            random_state=42,
            eval_metric="rmse",
            verbosity=0
        )

        # track learning curves
        eval_set = [(X_tr, y_tr[col]), (X_val, y_val[col])]
        reg.fit(
            X_tr,
            y_tr[col],
            eval_set=eval_set,
            verbose=False
        )
        train_time = time.time() - start_time
        print(f"    Training time: {train_time:.2f} s")

        # Metrics - learning and validation 
        results = reg.evals_result()
        train_rmse = results["validation_0"]["rmse"]
        val_rmse   = results["validation_1"]["rmse"]
        print(f"    Eval rounds: {len(train_rmse)}")
        print(f"    Train RMSE: first={train_rmse[0]:.4f}, last={train_rmse[-1]:.4f}")
        print(f"    Val   RMSE: first={val_rmse[0]:.4f}, last={val_rmse[-1]:.4f}")

        # Metrics - Cross validation and k-fold validation
        dtrain = xgb.DMatrix(X, y[col])
        params = reg.get_xgb_params()
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=reg.n_estimators,
            nfold=CV_FOLDS,
            metrics="rmse",
            seed=42,
            as_pandas=True,
            verbose_eval=False
        )
        mean_rmse = cv_results[f"test-rmse-mean"].iloc[-1]
        std_rmse  = cv_results[f"test-rmse-std"].iloc[-1]
        print(f"    {CV_FOLDS}-fold CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

        # Metrics - validation error
        y_val_pred = reg.predict(X_val)
        residuals  = y_val[col] - y_val_pred
        rmse_val   = np.sqrt(mean_squared_error(y_val[col], y_val_pred))
        variance   = np.var(residuals)
        max_error  = np.max(np.abs(residuals))
        print(
            f"    Val Metrics → RMSE: {rmse_val:.4f}, "
            f"Residual Var: {variance:.4f}, Max Err: {max_error:.4f}"
        )

        models[col] = reg

    return models

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
