"""
train_model.py
================
Script d'entraînement du modèle de détection de fraude bancaire.

Reprend fidèlement le pipeline développé dans le notebook
`Bank_Transaction_Scenario1.ipynb` :

1. Chargement des données (séparateur ";")
2. Nettoyage (suppression des identifiants, correction des noms de villes)
3. Feature engineering sur la date (Annee, Mois, Jour, Heure)
4. Traitement des valeurs extrêmes du Montant (log1p)
5. Encodage des variables catégorielles (LabelEncoder)
6. Split train/test stratifié
7. Standardisation (StandardScaler)
8. Rééquilibrage des classes (SMOTE)
9. Entraînement d'un RandomForestClassifier (hyperparamètres optimisés via
   GridSearchCV dans le notebook, réutilisables directement ici via --tune)
10. Évaluation (accuracy, classification_report, matrice de confusion)
11. Sauvegarde du modèle, du scaler, des encodeurs et de la liste des features

Utilisation :
    python model/train_model.py
    python model/train_model.py --data data/Bank_transaction_scenario1.csv --tune
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------------------------------------------------
# Colonnes à corriger : variantes d'orthographe / problèmes d'encodage
# repérés dans la variable "Localisation" pendant l'EDA du notebook.
# ----------------------------------------------------------------------
CORRECTION_VILLES = {
    "Saint Louis": "Saint-Louis",
    "Kafrine": "Kaffrine",
    "RunÃ©": "Rufisque",
}

ID_COLUMNS = ["ID Clients", "Numero de compte", "Identifiant operation"]

# Meilleurs hyperparamètres trouvés par GridSearchCV dans le notebook
# (scoring="f1_macro", cv=5). Utilisés par défaut pour aller vite ;
# relancer la recherche avec --tune si besoin de la reproduire.
BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}


def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV brut (séparateur ';')."""
    df = pd.read_csv(path, sep=";")
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage et feature engineering, comme dans le notebook."""
    df_clean = df.copy()

    # 1) Suppression des identifiants (pas de valeur prédictive)
    df_clean = df_clean.drop(columns=ID_COLUMNS)

    # 2) Date -> Annee / Mois / Jour / Heure, puis suppression de Date brute
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean["Annee"] = df_clean["Date"].dt.year
    df_clean["Mois"] = df_clean["Date"].dt.month
    df_clean["Jour"] = df_clean["Date"].dt.day
    df_clean["Heure"] = df_clean["Date"].dt.hour  # entier (0-23), plus utile qu'un objet time
    df_clean = df_clean.drop(columns=["Date"])

    # 3) Correction des variantes de villes
    df_clean["Localisation"] = df_clean["Localisation"].replace(CORRECTION_VILLES)

    # 4) Traitement des valeurs extrêmes de Montant via transformation log
    df_clean["Montant_log"] = np.log1p(df_clean["Montant"])

    return df_clean


def encode_categorical(df_clean: pd.DataFrame):
    """Encode toutes les colonnes catégorielles (y compris Target) avec LabelEncoder."""
    label_encoders = {}
    categorical_cols = df_clean.select_dtypes(include="object").columns

    for col in categorical_cols:
        encoder = LabelEncoder()
        df_clean[col] = encoder.fit_transform(df_clean[col])
        label_encoders[col] = encoder

    print("Colonnes encodées :", list(categorical_cols))
    if "Target" in label_encoders:
        classes = label_encoders["Target"].classes_
        print("Mapping Target ->", {i: c for i, c in enumerate(classes)})

    return df_clean, label_encoders


def train(args):
    df = load_data(args.data)
    df_clean = clean_data(df)
    df_clean, label_encoders = encode_categorical(df_clean)

    # Séparation features / cible
    X = df_clean.drop(columns=["Target"])
    y = df_clean["Target"]
    feature_columns = list(X.columns)
    print(f"Features utilisées ({len(feature_columns)}) : {feature_columns}")

    # Split stratifié train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardisation (fit sur train uniquement, pour éviter le data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Rééquilibrage des classes minoritaires (Fraude, Suspect) via SMOTE
    print("\nAvant SMOTE :")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print("\nAprès SMOTE :")
    print(y_train_res.value_counts())

    # Entraînement du modèle
    if args.tune:
        print("\nRecherche d'hyperparamètres (GridSearchCV)...")
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced"],
        }
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            scoring="f1_macro",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train_res, y_train_res)
        model = grid_search.best_estimator_
        print("Meilleurs hyperparamètres :", grid_search.best_params_)
        print("Meilleur score f1_macro (CV) :", grid_search.best_score_)
    else:
        print("\nEntraînement du RandomForestClassifier (hyperparamètres pré-optimisés)...")
        model = RandomForestClassifier(**BEST_PARAMS)
        model.fit(X_train_res, y_train_res)

    # Évaluation sur le jeu de test
    y_pred = model.predict(X_test_scaled)

    print("\n=== Rapport de classification ===")
    print(classification_report(y_test, y_pred))

    print("=== Matrice de confusion ===")
    print(confusion_matrix(y_test, y_pred))

    # Sauvegarde des artefacts nécessaires au déploiement (app.py)
    os.makedirs(args.output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(args.output_dir, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(args.output_dir, "label_encoders.pkl"))
    joblib.dump(feature_columns, os.path.join(args.output_dir, "feature_columns.pkl"))

    print(f"\nFichiers sauvegardés dans '{args.output_dir}/' :")
    print("  - fraud_model.pkl")
    print("  - scaler.pkl")
    print("  - label_encoders.pkl")
    print("  - feature_columns.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle de détection de fraude bancaire")
    parser.add_argument(
        "--data",
        type=str,
        default="data/Bank_transaction_scenario1.csv",
        help="Chemin vers le fichier CSV de transactions (séparateur ';')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model",
        help="Dossier de sortie pour le modèle et les artefacts",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Relance la recherche d'hyperparamètres (GridSearchCV) au lieu d'utiliser les valeurs pré-optimisées",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
