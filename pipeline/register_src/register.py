"""
Étape 4 du pipeline : Enregistrement du modèle

- Utilise mlflow.sklearn.save_model() en LOCAL (aucun appel réseau) pour
  packager le modèle au format MLflow standard (fichier MLmodel, conda.yaml,
  signature), puis mlflow.log_artifacts() pour l'envoyer au run.
- Enregistrement final via MlflowClient.create_model_version() (API bas
  niveau stable), en évitant mlflow.sklearn.log_model() et
  mlflow.register_model() qui déclenchent l'API "Logged Models" (MLflow 3.x)
  non supportée par le serveur de tracking Azure ML.
- N'enregistre le modèle QUE si le seuil de qualité (R²) est atteint.
- Déclenche ensuite le pipeline CD GitHub Actions (repository_dispatch).
"""
import argparse
import json
import os
import tempfile
import urllib.error
import urllib.request

import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


def trigger_github_cd(model_name, model_version, metrics):
    token = os.getenv("GH_DISPATCH_TOKEN")
    repo = os.getenv("GH_REPOSITORY")

    if not token or not repo:
        print("GH_DISPATCH_TOKEN ou GH_REPOSITORY absent : "
              "déclenchement du CD ignoré.")
        return

    url = f"https://api.github.com/repos/{repo}/dispatches"
    payload = {
        "event_type": "model-registered",
        "client_payload": {
            "model_name": model_name,
            "model_version": str(model_version),
            "rmse": round(metrics["rmse"], 2),
            "r2_score": round(metrics["r2_score"], 4),
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            print(f"Pipeline CD déclenché (HTTP {response.status}) "
                  f"pour {model_name} v{model_version}")
    except urllib.error.HTTPError as exc:
        print(f"Échec du déclenchement du CD : HTTP {exc.code} - {exc.read()}")
    except urllib.error.URLError as exc:
        print(f"Échec du déclenchement du CD : {exc.reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str)
    parser.add_argument("--metrics_input", type=str)
    parser.add_argument("--model_name", type=str, default="california-housing-model")
    parser.add_argument("--r2_threshold", type=float, default=0.7)
    args = parser.parse_args()

    with open(f"{args.metrics_input}/metrics.json") as f:
        metrics = json.load(f)

    print(f"Métriques reçues : {metrics}")

    if metrics["r2_score"] < args.r2_threshold:
        print(f"R² ({metrics['r2_score']:.4f}) sous le seuil "
              f"({args.r2_threshold}). Modèle REJETÉ, pas d'enregistrement.")
        return

    # Chargement du modèle entraîné (fichier joblib produit par train_sweep.py)
    model = joblib.load(f"{args.model_input}/model.pkl")

    with mlflow.start_run() as run:
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2_score", metrics["r2_score"])

        # --- Packaging au format MLflow standard, EN LOCAL (pas d'appel réseau) ---
        # Produit un dossier contenant : MLmodel, conda.yaml, python_env.yaml,
        # requirements.txt et le modèle sérialisé. C'est ce format que
        # l'US 2.1 du cahier des charges attend.
        local_model_dir = tempfile.mkdtemp()
        mlflow.sklearn.save_model(sk_model=model, path=f"{local_model_dir}/model")

        # --- Envoi du dossier complet vers le run (API stable, déjà éprouvée) ---
        mlflow.log_artifacts(f"{local_model_dir}/model", artifact_path="model")

        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    # Le "source" pointe maintenant vers le DOSSIER (contenant MLmodel),
    # pas vers un simple fichier .pkl comme dans la version précédente.
    model_source = f"{artifact_uri}/model"
    print(f"Source du modèle utilisée : {model_source}")

    client = MlflowClient()
    try:
        client.create_registered_model(args.model_name)
    except RestException:
        pass

    model_version = client.create_model_version(
        name=args.model_name,
        source=model_source,
        run_id=run_id,
    )

    print(f"Modèle enregistré (format MLflow) : {model_version.name}, "
          f"version {model_version.version}")

    trigger_github_cd(model_version.name, model_version.version, metrics)


if __name__ == "__main__":
    main()
