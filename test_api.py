import json
import os
import sys

import requests

# Même nom de variable que dans .github/workflows/ci-cd.yml (ENDPOINT_KEY / ENDPOINT_URI)
url = os.getenv(
    "ENDPOINT_URI",
    "https://california-housing-endpoint.eastus.inference.ml.azure.com/score",
)
api_key = os.getenv("ENDPOINT_KEY")

if not api_key:
    sys.exit(
        "Variable d'environnement ENDPOINT_KEY manquante.\n"
        "Récupérez la clé avec :\n"
        "  az ml online-endpoint get-credentials --name california-housing-endpoint "
        "--resource-group afenichamal --workspace-name ws-mlops-stage-amal "
        "--query primaryKey -o tsv"
    )

with open("sample-request.json", "r") as f:
    data = json.load(f)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

response = requests.post(url, headers=headers, json=data)

print("Status code :", response.status_code)
print("Response :", response.text)
