import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb

def get_numbers():
    print("Fetching data...")
    creditcard = fetch_openml(data_id=1597, as_frame=True)
    df = creditcard.frame
    df['Class'] = df['Class'].astype(str).replace({"'0'": 0, "'1'": 1, '0': 0, '1': 1, "0": 0, "1": 1}).astype(int)
    df['Time'] = df.index
    
    df["Hour"] = (df["Time"] % 86400) / 3600
    df["Log_Amount"] = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    
    CORE_FEATURES = ["V17", "V14", "V12", "V10", "V16", "V3", "V7", "V11", "V4", "V18", "V1", "V9", "V5", "V2", "V6", "Amount", "Time"]
    FEATURES = CORE_FEATURES + ["Hour", "Log_Amount", "Amount_zscore"]
    FEATURES = [f for f in FEATURES if f in df.columns]
    
    # ensure columns are float
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors='coerce')
        
    X = df[FEATURES]
    y = df["Class"]
    
    sample_size = 50000
    fraud_idx = df[df["Class"] == 1].index
    normal_idx = df[df["Class"] == 0].index
    n_normal = min(sample_size - len(fraud_idx), len(normal_idx))
    chosen_idx = list(fraud_idx) + list(np.random.default_rng(42).choice(normal_idx, size=n_normal, replace=False))
    
    X_s = X.loc[chosen_idx]
    y_s = y.loc[chosen_idx]
    
    sampler = SMOTE(random_state=42, k_neighbors=5)
    X_bal, y_bal = sampler.fit_resample(X_s, y_s)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_bal)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_bal, test_size=0.2, stratify=y_bal, random_state=42)
    imbalance_ratio = (y_s == 0).sum() / max((y_s == 1).sum(), 1)
    
    for model_name in ["XGBoost", "LightGBM"]:
        print(f"\n--- {model_name} ---")
        if model_name == "XGBoost":
            clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=imbalance_ratio, eval_metric="aucpr", random_state=42, verbosity=0)
        else:
            clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, is_unbalance=True, random_state=42, verbosity=-1)
        
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        
        best_f1 = 0
        best_metrics = {}
        for th in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_prob >= th).astype(int)
            f1 = f1_score(y_te, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    "accuracy": accuracy_score(y_te, y_pred),
                    "precision": precision_score(y_te, y_pred, zero_division=0),
                    "recall": recall_score(y_te, y_pred, zero_division=0),
                    "f1": f1,
                    "roc_auc": roc_auc_score(y_te, y_prob),
                    "avg_prec": average_precision_score(y_te, y_prob),
                }
        
        for k,v in best_metrics.items():
            print(f"{k}: {v:.4f}")

get_numbers()
