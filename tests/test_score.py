"""
Test unitaire du script de scoring, sans dépendance à Azure.
On simule un modèle factice pour tester uniquement la logique de run().
"""
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deployment"))

import score  # noqa: E402


class FakeModel:
    """Modèle factice qui renvoie toujours la même valeur, pour tester le format des I/O."""
    def predict(self, df):
        return np.array([100000.0] * len(df))


def test_run_returns_predictions_for_each_row():
    score.model = FakeModel()

    payload = {
        "data": [
            {"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
             "total_rooms": 880, "total_bedrooms": 129, "population": 322,
             "households": 126, "median_income": 8.3},
            {"longitude": -122.22, "latitude": 37.86, "housing_median_age": 21,
             "total_rooms": 7099, "total_bedrooms": 1106, "population": 2401,
             "households": 1138, "median_income": 8.3},
        ]
    }

    result = score.run(json.dumps(payload))

    assert "predictions" in result
    assert len(result["predictions"]) == 2
    assert all(isinstance(p, float) for p in result["predictions"])


def test_run_handles_invalid_input_gracefully():
    score.model = FakeModel()
    result = score.run("ceci n'est pas du json valide")
    assert "error" in result

