"""Train, compare and save the best churn model.

Usage:  python -m src.train [--data data/telco_churn.csv]
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import ROOT, load_config
from src.preprocessing import RAW_COLUMNS, TARGET, build_preprocessor, clean_target

RANDOM_STATE = 42


def get_candidates():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate(pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def train(data_path, model_out=None, metrics_out=None):
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {data_path}")
    df = clean_target(df.drop_duplicates())
    X = df[[c for c in RAW_COLUMNS if c in df.columns]]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, clf in get_candidates().items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        results[name] = evaluate(pipe, X_te, y_te)
        fitted[name] = pipe
        print(f"{name:20s} {results[name]}")

    best = max(results, key=lambda n: (results[n]["roc_auc"], results[n]["f1"]))
    print(f"\nBest model: {best}")

    cfg = load_config()
    model_out = Path(model_out or cfg["model_path"])
    metrics_out = Path(metrics_out or ROOT / "models" / "metrics.json")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], model_out)
    metrics_out.write_text(json.dumps(
        {"best_model": best, "selection_metric": "roc_auc", "results": results}, indent=2))
    print(f"Saved model   -> {model_out}\nSaved metrics -> {metrics_out}")
    return best, results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "telco_churn.csv"))
    args = ap.parse_args()
    train(args.data)
