import pandas as pd
import yaml
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    data = pd.read_csv(config["data"]["clean_data"])
    
    # создание новых признаков 
    data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
        "Mlle": "Rare", "Ms": "Rare", "Mme": "Rare", "Don": "Rare",
        "Lady": "Rare", "Sir": "Rare", "Capt": "Rare", "Countess": "Rare",
        "Jonkheer": "Rare"
    }
    data["Title"] = data["Title"].map(title_mapping).fillna("Rare")
    
    data["AgeGroup"] = pd.cut(data["Age"], bins=[0,12,19,60,100], labels=["Child","Teen","Adult","Senior"])
    data["MinorAge"] = (data["Age"] < 18).astype(int)
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    data["FareGroup"] = pd.qcut(data["Fare"], q=4, labels=["Low","Medium","High","VeryHigh"], duplicates="drop")
    
    data.drop(columns=["Name"], inplace=True, errors="ignore")
    
    # One‑hot encoding категориальных признаков
    categorical_features = config["features"]["categorical_features"]
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    bool_cols = data.select_dtypes(include="bool").columns
    data[bool_cols] = data[bool_cols].astype(int)
    
    target = config["features"]["target_col"]
    X = data.drop(columns=[target])
    y = data[target]
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["features"]["test_size"],
        random_state=config["features"]["random_state"],
        stratify=y
    )
    
    # масштабирование числовых признаков
    numeric_features = config["features"]["numeric_features"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # Сохранение готовых данных для обучения и тестирования
    X_train_scaled.to_csv("data/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv("data/X_test_scaled.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    
    # обученный scaler 
    joblib.dump(scaler, "models/scaler.pkl")
    
    print('признаки сгенерированы, данные сохранены')

if __name__ == "__main__":
    main()