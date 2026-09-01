"""
Étape 1 du pipeline : Préparation des données
- Nettoyage (traitement des valeurs manquantes)
- Feature engineering simple
- Split Train/Validation
"""
import argparse
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage + feature engineering, extrait en fonction pure pour être testable
    sans dépendance à Azure ni au système de fichiers."""
    df = df.copy()

    # Nettoyage : traitement des valeurs manquantes sur total_bedrooms
    median_bedrooms = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(median_bedrooms)

    # Feature engineering simple (ratios utiles pour ce dataset)
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    # Encodage de la variable catégorielle ocean_proximity
    if "ocean_proximity" in df.columns:
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Chemin vers le dossier contenant le dataset brut (Data Asset)")
    parser.add_argument("--train_data", type=str, help="Chemin de sortie pour les données d'entraînement")
    parser.add_argument("--val_data", type=str, help="Chemin de sortie pour les données de validation")
    args = parser.parse_args()

    # 1. Chargement des données brutes : le Data Asset est un dossier (uri_folder),
    #    on cherche automatiquement le premier fichier CSV qu'il contient.
    csv_files = glob.glob(os.path.join(args.input_data, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {args.input_data}")
    df = pd.read_csv(csv_files[0])
    print(f"Fichier chargé : {csv_files[0]}")
    print(f"Données brutes chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    n_missing = df["total_bedrooms"].isna().sum()
    print(f"Valeurs manquantes détectées sur total_bedrooms : {n_missing}")

    # 2. Nettoyage + feature engineering (fonction testable)
    df = clean_and_engineer(df)

    # 3. Split Train / Validation (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train : {train_df.shape[0]} lignes | Validation : {val_df.shape[0]} lignes")

    # 4. Sauvegarde des sorties (transmises automatiquement à l'étape suivante du pipeline)
    train_df.to_csv(f"{args.train_data}/train.csv", index=False)
    val_df.to_csv(f"{args.val_data}/val.csv", index=False)
    print("Préparation des données terminée.")


if __name__ == "__main__":
    main()
