"""
Script de scoring exécuté par l'Azure Managed Online Endpoint.
Deux fonctions attendues par le serveur d'inférence : init() et run().

Pour l'A/B testing, chaque déploiement (champion/challenger) définit la variable
d'environnement DEPLOYMENT_NAME dans son fichier YAML. Cela permet de savoir,
dans les logs et dans la réponse, quelle version du modèle a traité la requête.
"""
import os
import glob
import json
import joblib
import pandas as pd

model = None
deployment_name = None


def init():
    """Appelé une seule fois au démarrage du conteneur : charge le modèle en mémoire."""
    global model, deployment_name
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    deployment_name = os.getenv("DEPLOYMENT_NAME", "unknown")

    # Recherche du fichier model.pkl où qu'il soit dans le dossier du modèle monté
    candidates = glob.glob(os.path.join(model_dir, "**", "model.pkl"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"Aucun model.pkl trouvé dans {model_dir}")

    model = joblib.load(candidates[0])
    print(f"[{deployment_name}] Modèle chargé depuis : {candidates[0]}")


def run(raw_data):
    """Appelé à chaque requête reçue par l'endpoint.

    Format d'entrée attendu (JSON) :
    {"data": [{"longitude": -122.23, "latitude": 37.88, ...}, ...]}
    """
    try:
        payload = json.loads(raw_data)
        df = pd.DataFrame(payload["data"])
        predictions = model.predict(df)

        # Log explicite pour pouvoir distinguer Champion/Challenger dans les logs d'inférence
        print(f"[{deployment_name}] {len(df)} prédiction(s) servie(s)")

        return {
            "predictions": predictions.tolist(),
            "deployment": deployment_name,
        }
    except Exception as e:
        print(f"[{deployment_name}] Erreur : {e}")
        return {"error": str(e), "deployment": deployment_name}
