"""
Étape 4 du pipeline : Enregistrement du modèle

- Utilise les API bas niveau et stables de MLflow (log_artifact,
  MlflowClient.create_model_version) pour éviter le chemin "Logged Models"
  non supporté par le tracking server Azure ML.
- N'enregistre le modèle QUE si le seuil de qualité (R²) est atteint.
- Déclenche ensuite le pipeline CD GitHub Actions (repository_dispatch),
  ce qui satisfait le critère "CD déclenché par l'enregistrement d'un
  nouveau modèle" du cahier des charges.
"""
import argparse
import json
import os
import urllib.error
import urllib.request

import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


def trigger_github_cd(model_name, model_version, metrics):
    """Notifie GitHub Actions qu'un nouveau modèle vient d'être enregistré.

    Utilise l'événement repository_dispatch de l'API GitHub. Le token et le
    dépôt sont lus depuis les variables d'environnement du job (par leur NOM,
    pas par leur valeur) ; si elles sont absentes, on n'échoue pas le
    pipeline, on se contente d'avertir.
    """
    token = os.getenv("GH_DISPATCH_TOKEN")
    repo = os.getenv("GH_REPOSITORY")

    # --- DIAGNOSTIC TEMPORAIRE (à retirer une fois le problème résolu) ---
    print(f"[DEBUG] GH_REPOSITORY reçu : {repo!r}")
    if not token:
        print("[DEBUG] GH_DISPATCH_TOKEN reçu : (vide ou absent)")
    else:
        print(f"[DEBUG] GH_DISPATCH_TOKEN reçu : {len(token)} caractères, "
              f"commence par {token[:12]}...")
    # --- Fin du diagnostic ---

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

    with mlflow.start_run() as run:
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2_score", metrics["r2_score"])
        mlflow.log_artifact(f"{args.model_input}/model.pkl", artifact_path="model")
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    model_source = f"{artifact_uri}/model/model.pkl"
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

    print(f"Modèle enregistré : {model_version.name}, "
          f"version {model_version.version}")

    trigger_github_cd(model_version.name, model_version.version, metrics)


if __name__ == "__main__":
    main()
