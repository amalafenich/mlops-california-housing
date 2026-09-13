import requests
import json

url = "https://california-housing-endpoint.eastus.inference.ml.azure.com/score"
api_key = "5POPuJJ01irtnfguJ53qNYki6n0WaenrZr3OSoplCgA3ZzSaO1x6JQQJ99CIAAAAAAAAAAAAINFRAZML47E3"

with open("sample-request.json", "r") as f:
    data = json.load(f)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.post(
    url,
    headers=headers,
    json=data
)

print("Status code :", response.status_code)
print("Response :", response.text)