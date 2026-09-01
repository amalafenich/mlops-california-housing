"""
Test unitaire de la logique de nettoyage des données (clean_and_engineer),
sans dépendance à Azure ni à un vrai fichier de données.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "prep_src"))

from prep import clean_and_engineer  # noqa: E402


def make_sample_df():
    return pd.DataFrame({
        "total_rooms": [1000, 2000, 1500],
        "total_bedrooms": [200, None, 300],
        "households": [100, 200, 150],
        "population": [500, 800, 600],
        "median_house_value": [200000, 300000, 250000],
        "ocean_proximity": ["NEAR BAY", "INLAND", "NEAR BAY"],
    })


def test_missing_values_are_filled():
    df = make_sample_df()
    result = clean_and_engineer(df)
    assert result["total_bedrooms"].isna().sum() == 0


def test_missing_value_filled_with_median():
    df = make_sample_df()
    expected_median = df["total_bedrooms"].median()  # médiane calculée sur les valeurs non manquantes
    result = clean_and_engineer(df)
    # La ligne qui avait une valeur manquante doit être remplacée par la médiane
    assert result.loc[1, "total_bedrooms"] == expected_median


def test_engineered_features_are_created():
    df = make_sample_df()
    result = clean_and_engineer(df)
    assert "rooms_per_household" in result.columns
    assert "bedrooms_per_room" in result.columns
    assert "population_per_household" in result.columns


def test_categorical_column_is_encoded():
    df = make_sample_df()
    result = clean_and_engineer(df)
    assert "ocean_proximity" not in result.columns
    # Au moins une colonne encodée (one-hot) doit apparaître
    assert any(col.startswith("ocean_proximity_") for col in result.columns)
