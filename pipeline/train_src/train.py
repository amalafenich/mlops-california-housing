"""
Étape 2 du pipeline : Entraînement du modèle
- Entraînement d'un RandomForestRegressor
- Tracking automatique des paramètres avec MLflow
"""
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Chemin vers les données d'entraînement")
    parser.add_argument("--n_estimators", type=int, default=200, help="Nombre d'arbres")
    parser.add_argument("--max_depth", type=int, default=15, help="Profondeur maximale")
    parser.add_argument("--model_output", type=str, help="Chemin de sortie du modèle entraîné")
    args = parser.parse_args()

    # Active le tracking automatique MLflow (paramètres, métriques d'entraînement)
    mlflow.sklearn.autolog()

    # 1. Chargement des données d'entraînement préparées
    train_df = pd.read_csv(f"{args.train_data}/train.csv")
    target_col = "median_house_value"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print(f"Entraînement sur {X_train.shape[0]} lignes, {X_train.shape[1]} features")

    # 2. Entraînement du modèle
    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # 3. Sauvegarde du modèle pour l'étape suivante du pipeline
        os.makedirs(args.model_output, exist_ok=True)
        joblib.dump(model, f"{args.model_output}/model.pkl")
        # Sauvegarde aussi la liste des colonnes utilisées (utile à l'évaluation)
        X_train.columns.to_series().to_csv(f"{args.model_output}/columns.csv", index=False)

    print("Entraînement terminé, modèle sauvegardé.")


if __name__ == "__main__":
    main()
