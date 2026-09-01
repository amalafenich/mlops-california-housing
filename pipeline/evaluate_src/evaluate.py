"""
Étape 3 du pipeline : Évaluation du modèle
- Calcul des métriques (RMSE, R²) sur le jeu de validation
- Log des métriques avec MLflow
"""
import argparse
import pandas as pd
import joblib
import json
import mlflow
from sklearn.metrics import mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data", type=str, help="Chemin vers les données de validation")
    parser.add_argument("--model_input", type=str, help="Chemin vers le modèle entraîné")
    parser.add_argument("--metrics_output", type=str, help="Chemin de sortie des métriques")
    args = parser.parse_args()

    # 1. Chargement du modèle et des données de validation
    model = joblib.load(f"{args.model_input}/model.pkl")
    val_df = pd.read_csv(f"{args.val_data}/val.csv")

    target_col = "median_house_value"
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # 2. Prédiction et calcul des métriques
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)

    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

    # 3. Sauvegarde des métriques pour l'étape d'enregistrement (décision automatique)
    metrics = {"rmse": rmse, "r2_score": r2}
    with open(f"{args.metrics_output}/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Évaluation terminée.")


if __name__ == "__main__":
    main()
