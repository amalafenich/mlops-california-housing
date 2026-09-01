"""
Script de scoring exécuté par l'Azure Managed Online Endpoint.
Deux fonctions attendues par le serveur d'inférence : init() et run().
"""
import os
import glob
import json
import joblib
import pandas as pd

model = None


def init():
    """Appelé une seule fois au démarrage du conteneur : charge le modèle en mémoire."""
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")

    # Recherche du fichier model.pkl où qu'il soit dans le dossier du modèle monté
    candidates = glob.glob(os.path.join(model_dir, "**", "model.pkl"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"Aucun model.pkl trouvé dans {model_dir}")

    model = joblib.load(candidates[0])
    print(f"Modèle chargé depuis : {candidates[0]}")


def run(raw_data):
    """Appelé à chaque requête reçue par l'endpoint.

    Format d'entrée attendu (JSON) :
    {"data": [{"longitude": -122.23, "latitude": 37.88, ...}, ...]}
    """
    try:
        payload = json.loads(raw_data)
        df = pd.DataFrame(payload["data"])
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
