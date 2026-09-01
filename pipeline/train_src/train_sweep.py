"""
Étape 2 du pipeline (version sweep) : Entraînement + évaluation rapide
- Entraîne un RandomForestRegressor avec les hyperparamètres reçus (un essai du sweep)
- Évalue immédiatement sur le jeu de validation et logge le R² comme métrique cible,
  utilisée par Azure ML pour choisir le meilleur essai du sweep.
"""
import argparse
import os
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Chemin vers les données d'entraînement")
    parser.add_argument("--val_data", type=str, help="Chemin vers les données de validation")
    parser.add_argument("--n_estimators", type=int, default=200, help="Nombre d'arbres")
    parser.add_argument("--max_depth", type=int, default=15, help="Profondeur maximale")
    parser.add_argument("--model_output", type=str, help="Chemin de sortie du modèle entraîné")
    args = parser.parse_args()

    # 1. Chargement des données
    train_df = pd.read_csv(f"{args.train_data}/train.csv")
    val_df = pd.read_csv(f"{args.val_data}/val.csv")
    target_col = "median_house_value"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    print(f"Essai : n_estimators={args.n_estimators}, max_depth={args.max_depth}")

    # 2. Entraînement
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 3. Évaluation rapide sur le jeu de validation
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    print(f"RMSE={rmse:.2f} | R2={r2:.4f}")

    # 4. Log des paramètres et de la métrique cible pour le sweep
    #    (le nom "r2_score" doit correspondre au primary_metric défini dans le pipeline)
    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

    # 5. Sauvegarde du modèle de cet essai
    os.makedirs(args.model_output, exist_ok=True)
    joblib.dump(model, f"{args.model_output}/model.pkl")
    X_train.columns.to_series().to_csv(f"{args.model_output}/columns.csv", index=False)

    print("Essai terminé.")


if __name__ == "__main__":
    main()
