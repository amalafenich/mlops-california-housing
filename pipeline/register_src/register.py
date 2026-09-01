"""
Étape 4 du pipeline : Enregistrement du modèle
- Utilise uniquement les API bas niveau et stables de MLflow (log_artifact,
  MlflowClient.create_model_version) pour éviter le nouveau chemin "Logged Models"
  (mlflow.sklearn.log_model/save_model) non supporté par le tracking server Azure ML.
- Aucune installation de package nécessaire.
- N'enregistre le modèle QUE si le seuil de qualité (R²) est atteint.
"""
import argparse
import json
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, help="Chemin vers le modèle entraîné")
    parser.add_argument("--metrics_input", type=str, help="Chemin vers les métriques d'évaluation")
    parser.add_argument("--model_name", type=str, default="california-housing-model")
    parser.add_argument("--r2_threshold", type=float, default=0.7, help="Seuil minimal de R² pour valider le modèle")
    args = parser.parse_args()

    # 1. Lecture des métriques calculées à l'étape précédente
    with open(f"{args.metrics_input}/metrics.json") as f:
        metrics = json.load(f)

    print(f"Métriques reçues : {metrics}")

    # 2. Décision automatique : seuil de qualité
    if metrics["r2_score"] < args.r2_threshold:
        print(f"R² ({metrics['r2_score']:.4f}) sous le seuil ({args.r2_threshold}). "
              f"Modèle REJETÉ, pas d'enregistrement.")
        return

    # 3. Log du fichier modèle comme simple artefact (pas de flavor MLflow, pas de Model.log())
    with mlflow.start_run() as run:
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2_score", metrics["r2_score"])
        mlflow.log_artifact(f"{args.model_input}/model.pkl", artifact_path="model")
        run_id = run.info.run_id
        # URI native générée par Azure ML (format azureml://artifacts/...),
        # à ne pas confondre avec la notation abstraite "runs:/..." de MLflow
        artifact_uri = run.info.artifact_uri

    # 4. Enregistrement explicite via l'API bas niveau, avec la bonne URI native
    model_source = f"{artifact_uri}/model/model.pkl"
    print(f"Source du modèle utilisée : {model_source}")

    client = MlflowClient()
    try:
        client.create_registered_model(args.model_name)
    except RestException:
        pass  # le modèle enregistré existe déjà, on ajoute juste une nouvelle version

    model_version = client.create_model_version(
        name=args.model_name,
        source=model_source,
        run_id=run_id,
    )

    print(f"Modèle enregistré : {model_version.name}, version {model_version.version}")


if __name__ == "__main__":
    main()
