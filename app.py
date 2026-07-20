"""
app.py
======
Application Streamlit de détection de fraude bancaire.

Charge les artefacts produits par `model/train_model.py` :
    - model/fraud_model.pkl      (RandomForestClassifier)
    - model/scaler.pkl           (StandardScaler)
    - model/label_encoders.pkl   (dict de LabelEncoder par colonne catégorielle)
    - model/feature_columns.pkl  (liste ordonnée des colonnes attendues par le modèle)

Fonctionnalités :
    1. Analyse d'une transaction saisie manuellement
    2. Analyse d'un fichier CSV contenant plusieurs transactions
    3. Téléchargement des résultats au format CSV
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "model"

CORRECTION_VILLES = {
    "Saint Louis": "Saint-Louis",
    "Kafrine": "Kaffrine",
    "RunÃ©": "Rufisque",
}

# Couleurs / icônes pour l'affichage des résultats
CLASS_STYLE = {
    "Normal": {"emoji": "🟢", "color": "green"},
    "Suspect": {"emoji": "🟠", "color": "orange"},
    "Fraude": {"emoji": "🔴", "color": "red"},
}


# ----------------------------------------------------------------------
# Chargement des artefacts (mis en cache pour ne pas recharger à chaque interaction)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return model, scaler, label_encoders, feature_columns


def safe_encode(encoder, value):
    """Encode une valeur catégorielle avec un LabelEncoder déjà entraîné.
    Si la valeur n'a jamais été vue à l'entraînement, on retombe sur la
    classe la plus fréquente plutôt que de planter l'application."""
    value = str(value)
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    st.warning(
        f"Valeur inconnue '{value}' pour une variable catégorielle : "
        "remplacée par la catégorie la plus proche disponible."
    )
    return encoder.transform([encoder.classes_[0]])[0]


def preprocess(df_raw: pd.DataFrame, label_encoders: dict, feature_columns: list) -> pd.DataFrame:
    """Applique exactement le même pipeline de prétraitement que train_model.py,
    mais en *transform* uniquement (aucun ré-entraînement des encodeurs/scaler)."""
    df = df_raw.copy()

    # Colonnes identifiants à ignorer si présentes dans le CSV importé
    for col in ["ID Clients", "Numero de compte", "Identifiant operation", "Target"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Date -> Annee / Mois / Jour / Heure
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Annee"] = df["Date"].dt.year
        df["Mois"] = df["Date"].dt.month
        df["Jour"] = df["Date"].dt.day
        df["Heure"] = df["Date"].dt.hour
        df = df.drop(columns=["Date"])

    # Correction des variantes de villes
    if "Localisation" in df.columns:
        df["Localisation"] = df["Localisation"].replace(CORRECTION_VILLES)

    # Transformation log du montant
    if "Montant" in df.columns:
        df["Montant_log"] = np.log1p(df["Montant"])

    # Encodage des variables catégorielles avec les encodeurs sauvegardés
    for col, encoder in label_encoders.items():
        if col == "Target":
            continue
        if col in df.columns:
            df[col] = df[col].apply(lambda v: safe_encode(encoder, v))

    # Réordonner / compléter les colonnes pour correspondre exactement à l'entraînement
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # valeur par défaut si une colonne attendue est absente

    df = df[feature_columns]
    return df


def predict(df_processed: pd.DataFrame, model, scaler, target_encoder):
    X_scaled = scaler.transform(df_processed)
    preds = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)

    labels = target_encoder.inverse_transform(preds)
    proba_df = pd.DataFrame(probas, columns=target_encoder.inverse_transform(model.classes_))
    return labels, proba_df


# ----------------------------------------------------------------------
# Interface Streamlit
# ----------------------------------------------------------------------
st.set_page_config(page_title="Détection de Fraude Bancaire", page_icon="", layout="wide")

st.title(" Détection de Fraude Bancaire avec Machine Learning")
st.write(
    "Cette application utilise un modèle **Random Forest** entraîné pour classer "
    "les transactions bancaires en trois catégories : **Normal**, **Suspect**, **Fraude**."
)

try:
    model, scaler, label_encoders, feature_columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "Impossible de trouver les fichiers du modèle dans le dossier `model/`. "
        "Entraîne d'abord le modèle avec `python model/train_model.py`."
    )
    st.stop()

target_encoder = label_encoders["Target"]
localisations_connues = sorted(label_encoders["Localisation"].classes_) if "Localisation" in label_encoders else []
types_connus = sorted(label_encoders["Type de transaction"].classes_) if "Type de transaction" in label_encoders else []
statuts_connus = sorted(label_encoders["Status operation"].classes_) if "Status operation" in label_encoders else []

tab1, tab2 = st.tabs(["🔎 Analyser une transaction", " Analyser un fichier CSV"])

# ------------------------------------------------------------------
# Onglet 1 : transaction unique
# ------------------------------------------------------------------
with tab1:
    st.subheader("Renseigner les caractéristiques de la transaction")

    col1, col2 = st.columns(2)

    with col1:
        montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
        type_transaction = st.selectbox("Type de transaction", types_connus) if types_connus else st.text_input("Type de transaction")
        localisation = st.selectbox("Localisation", localisations_connues) if localisations_connues else st.text_input("Localisation")

    with col2:
        statut = st.selectbox("Statut de l'opération", statuts_connus) if statuts_connus else st.text_input("Statut de l'opération")
        date_transaction = st.date_input("Date de la transaction")
        heure_transaction = st.time_input("Heure de la transaction")

    if st.button("Analyser la transaction", type="primary"):
        date_complete = pd.Timestamp.combine(date_transaction, heure_transaction)

        transaction = pd.DataFrame([{
            "Date": date_complete,
            "Montant": montant,
            "Type de transaction": type_transaction,
            "Localisation": localisation,
            "Status operation": statut,
        }])

        df_processed = preprocess(transaction, label_encoders, feature_columns)
        labels, proba_df = predict(df_processed, model, scaler, target_encoder)

        classe_predite = labels[0]
        style = CLASS_STYLE.get(classe_predite, {"emoji": "⚪", "color": "gray"})

        st.markdown(f"### Résultat : {style['emoji']} :{style['color']}[{classe_predite}]")

        st.write("**Probabilités par classe :**")
        st.bar_chart(proba_df.T.rename(columns={0: "Probabilité"}))
        st.dataframe(proba_df.style.format("{:.2%}"))

# ------------------------------------------------------------------
# Onglet 2 : fichier CSV
# ------------------------------------------------------------------
with tab2:
    st.subheader("Importer un fichier CSV de transactions")
    st.caption(
        "Le fichier doit contenir au minimum les colonnes : "
        "`Date`, `Montant`, `Type de transaction`, `Localisation`, `Status operation`."
    )

    fichier = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            df_input = pd.read_csv(fichier, sep=";")
            if df_input.shape[1] == 1:
                # Le séparateur n'était probablement pas ";"
                fichier.seek(0)
                df_input = pd.read_csv(fichier)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            st.stop()

        st.write("Aperçu des données importées :")
        st.dataframe(df_input.head())

        if st.button("Lancer l'analyse du fichier", type="primary"):
            df_processed = preprocess(df_input, label_encoders, feature_columns)
            labels, proba_df = predict(df_processed, model, scaler, target_encoder)

            df_resultats = df_input.copy()
            df_resultats["Prediction"] = labels
            for classe in proba_df.columns:
                df_resultats[f"Proba_{classe}"] = proba_df[classe].values

            st.success(f"{len(df_resultats)} transactions analysées.")

            st.write("**Répartition des prédictions :**")
            st.bar_chart(df_resultats["Prediction"].value_counts())

            st.write("**Résultats détaillés :**")
            st.dataframe(df_resultats)

            csv_export = df_resultats.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv_export,
                file_name="resultats_detection_fraude.csv",
                mime="text/csv",
            )

st.divider()
st.caption("Projet académique — Master 1 Intelligence Artificielle (Big Data & IA) — Bousso Seck")
