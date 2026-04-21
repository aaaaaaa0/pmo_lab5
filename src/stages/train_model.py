# src/stages/train_model.py
import pandas as pd
import yaml
import joblib
from sklearn.linear_model import LogisticRegression

def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    X_train = pd.read_csv("data/X_train_scaled.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()  # превращаем в 1D
    
    params = config["train"]["params"]
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Сохраняем модель
    joblib.dump(model, "models/logistic_model.pkl")
    print("Модель обучена и сохранена.")

if __name__ == "__main__":
    main()