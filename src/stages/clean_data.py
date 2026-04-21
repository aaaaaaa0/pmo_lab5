# src/stages/clean_data.py
import pandas as pd
import yaml
from scipy.stats.mstats import winsorize

def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv(config["data"]["raw_data"])
    data = df.copy()
    
    # 1. Удаление ненужных столбцов
    data.drop(columns=config["features"]["drop_cols"], inplace=True, errors="ignore")
    
    # 2. Обработка пропусков
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["Age"] = data.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))
    
    # 3. Обработка выбросов (winsorize)
    data["Fare"] = winsorize(data["Fare"].to_numpy(), limits=(0, 0.01))
    
    # Сохраняем очищенный датасет
    data.to_csv(config["data"]["clean_data"], index=False)
    print(f"Очищенные данные сохранены в {config['data']['clean_data']}")

if __name__ == "__main__":
    main()