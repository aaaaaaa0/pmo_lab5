# src/stages/evaluate_model.py
import pandas as pd
import yaml
import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import dvclive

def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    X_test = pd.read_csv("data/X_test_scaled.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    
    model = joblib.load("models/logistic_model.pkl")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba)
    }
    
    # Сохраняем как JSON
    with open(config["evaluate"]["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Логируем через dvclive (для графиков)
    for name, val in metrics.items():
        dvclive.log(name, val)
    
    print("Метрики:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()