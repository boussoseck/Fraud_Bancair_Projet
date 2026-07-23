import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# ==================================================
# 1. CONFIGURATION GÉNÉRALE
# ==================================================

st.set_page_config(
    page_title="Détection de fraude bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

COULEUR_NORMAL = "#2A9D8F"
COULEUR_SUSPECT = "#F4A261"
COULEUR_FRAUDE = "#E63946"
COULEUR_PRIMAIRE = "#1B2A4A"

PALETTE_TARGET = {
    "Normal": COULEUR_NORMAL,
    "Suspect": COULEUR_SUSPECT,
    "Fraude": COULEUR_FRAUDE
}

# Métriques du modèle, mesurées lors de l'entraînement (voir notebook).
# À remplacer si vous sauvegardez ces valeurs dans un fichier lors du training.
METRIQUES_MODELE = {
    "accuracy": 0.880,
    "precision": 0.882,
    "recall": 0.880,
    "f1": 0.881,
    "par_classe": {
        "Normal":  {"precision": 0.93, "recall": 0.93, "f1": 0.93, "support": 819},
        "Suspect": {"precision": 0.74, "recall": 0.74, "f1": 0.74, "support": 218},
        "Fraude":  {"precision": 0.60, "recall": 0.68, "f1": 0.64, "support": 40},
    },
    "matrice_confusion": {
        "labels": ["Fraude", "Normal", "Suspect"],
        "valeurs": [
            [27, 8, 5],
            [4, 762, 53],
            [14, 43, 161]
        ]
    }
}

CHEMIN_DONNEES_HISTORIQUES = "data/Bank_transaction_scenario.csv"
CHEMIN_MODELE = "model/fraud_model.pkl"
CHEMIN_SCALER = "model/scaler.pkl"
CHEMIN_ENCODEURS = "model/label_encoders.pkl"
CHEMIN_COLONNES = "model/feature_columns.pkl"


# ==================================================
# 2. CHARGEMENT DU MODÈLE ET DES DONNÉES
# ==================================================
#
# Le notebook sauvegarde 4 artefacts distincts (pas un Pipeline scikit-learn) :
#   - fraud_model.pkl      -> RandomForestClassifier entraîné
#   - scaler.pkl           -> StandardScaler ajusté sur X_train
#   - label_encoders.pkl   -> dict {nom_colonne: LabelEncoder}, une entrée par
#                             colonne catégorielle ENCODÉE PENDANT L'ENTRAÎNEMENT,
#                             y compris "Target" et "Heure"
#   - feature_columns.pkl  -> liste ordonnée des colonnes attendues par le modèle
#
# On recharge donc les 4 fichiers et on reproduit exactement le même
# prétraitement que dans le notebook avant d'appeler le modèle.

@st.cache_resource
def charger_artefacts():
    modele = joblib.load(CHEMIN_MODELE)
    scaler = joblib.load(CHEMIN_SCALER)
    label_encoders = joblib.load(CHEMIN_ENCODEURS)
    feature_columns = joblib.load(CHEMIN_COLONNES)
    return modele, scaler, label_encoders, feature_columns


@st.cache_data
def charger_donnees_historiques():
    """Charge le jeu de données historique utilisé pour l'entraînement,
    pour alimenter le tableau de bord. Retourne None si le fichier est absent."""
    if not os.path.exists(CHEMIN_DONNEES_HISTORIQUES):
        return None
    df = pd.read_csv(CHEMIN_DONNEES_HISTORIQUES, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


modele, scaler, label_encoders, feature_columns = charger_artefacts()
df_historique = charger_donnees_historiques()

localisations = (
    sorted(label_encoders["Localisation"].classes_.tolist())
    if "Localisation" in label_encoders else []
)
types_transaction_options = (
    sorted(label_encoders["Type de transaction"].classes_.tolist())
    if "Type de transaction" in label_encoders else ["ATM", "Paiement en ligne", "Paiement électronique"]
)
status_options = (
    sorted(label_encoders["Status operation"].classes_.tolist())
    if "Status operation" in label_encoders else ["Validé", "Echoué", "En attente"]
)


# ==================================================
# 3. FONCTIONS UTILITAIRES
# ==================================================

def encoder_valeur(nom_colonne, valeur):
    """Encode une valeur catégorielle avec le LabelEncoder sauvegardé.
    Si la valeur n'a jamais été vue à l'entraînement, retourne -1 au lieu
    de planter (utile pour "Heure", quasi unique par transaction, ou une
    localisation absente du jeu d'entraînement)."""
    encodeur = label_encoders.get(nom_colonne)
    if encodeur is None:
        return valeur
    if valeur in encodeur.classes_:
        return int(encodeur.transform([valeur])[0])
    return -1

def construire_ligne_modele(type_transaction, status_operation, localisation,
                             montant, date_heure):

    valeurs = {
        "Type de transaction": encoder_valeur("Type de transaction", type_transaction),
        "Status operation": encoder_valeur("Status operation", status_operation),
        "Localisation": encoder_valeur("Localisation", localisation),
        "Montant": montant,
        "Annee": date_heure.year,
        "Mois": date_heure.month,
        "Jour": date_heure.day,
        "Heure": date_heure.hour,
        "Montant_log": np.log1p(montant),
    }

    ligne = pd.DataFrame([valeurs])

    for colonne in feature_columns:
        if colonne not in ligne.columns:
            ligne[colonne] = 0

    return ligne[feature_columns]


def preparer_donnees(df):
    """Prépare un DataFrame brut (issu d'un CSV importé) pour la prédiction,
    en reproduisant le feature engineering du notebook."""

    donnees = df.copy()

    corrections_localisation = {
        "Saint-Louis": "Saint Louis",
        "Kafrine": "Kaffrine",
        "RunÃ©": "Runé"
    }
    donnees["Localisation"] = donnees["Localisation"].replace(corrections_localisation)

    donnees["Date"] = pd.to_datetime(donnees["Date"], errors="coerce")
    donnees["Annee"] = donnees["Date"].dt.year
    donnees["Mois"] = donnees["Date"].dt.month
    donnees["Jour"] = donnees["Date"].dt.day
    donnees["Heure"] = donnees["Date"].dt.hour
    donnees["Montant_log"] = np.log1p(donnees["Montant"])

    for colonne in ["Type de transaction", "Status operation", "Localisation"]:
        if colonne in label_encoders:
            donnees[colonne] = donnees[colonne].apply(lambda v, c=colonne: encoder_valeur(c, v))

    for colonne in feature_columns:
        if colonne not in donnees.columns:
            donnees[colonne] = 0

    return donnees[feature_columns]


def predire(X):
    """Standardise puis prédit ; renvoie les classes texte (Normal/Suspect/Fraude)
    et les probabilités associées, même si le modèle a été entraîné sur un Target
    encodé en entiers (0/1/2)."""

    X_scaled = scaler.transform(X)
    predictions_brutes = modele.predict(X_scaled)
    probabilites = modele.predict_proba(X_scaled)

    if "Target" in label_encoders:
        encodeur_target = label_encoders["Target"]
        predictions = encodeur_target.inverse_transform(predictions_brutes.astype(int))
        classes = encodeur_target.inverse_transform(modele.classes_.astype(int))
    else:
        predictions = predictions_brutes
        classes = modele.classes_

    return predictions, probabilites, classes


def carte_kpi(colonne, titre, valeur, sous_titre=None, couleur=COULEUR_PRIMAIRE):
    """Affiche une carte KPI stylisée dans une colonne Streamlit donnée."""
    with colonne:
        st.markdown(
            f"""
            <div style="
                background-color:white;
                border:1px solid #E5E9F0;
                border-radius:12px;
                padding:18px 20px;
                box-shadow:0 2px 6px rgba(0,0,0,0.05);
            ">
                <div style="color:#6B7280; font-size:13px; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.5px;">
                    {titre}
                </div>
                <div style="color:{couleur}; font-size:32px; font-weight:700; margin-top:4px;">
                    {valeur}
                </div>
                <div style="color:#9CA3AF; font-size:12px; margin-top:2px;">
                    {sous_titre or ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def jauge_risque(probabilite_fraude):
    """Construit une jauge Plotly représentant le niveau de risque d'une transaction."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilite_fraude * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Probabilité de fraude", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": COULEUR_PRIMAIRE},
            "steps": [
                {"range": [0, 30], "color": "#D8F0EC"},
                {"range": [30, 65], "color": "#FCE7CF"},
                {"range": [65, 100], "color": "#FAD4D7"},
            ],
            "threshold": {
                "line": {"color": COULEUR_FRAUDE, "width": 4},
                "thickness": 0.8,
                "value": probabilite_fraude * 100
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ==================================================
# 4. BARRE LATÉRALE — NAVIGATION
# ==================================================

st.sidebar.markdown("## 🏦 Détection de fraude")
st.sidebar.caption("Application de scoring des transactions bancaires")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Tableau de bord",
        "🔍 Transaction unique",
        "📁 Fichier CSV",
        "🧪 Performance du modèle",
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.info(
    "Application pédagogique réalisée avec "
    "Python, scikit-learn et Streamlit."
)


# ==================================================
# 5. PAGE — TABLEAU DE BORD
# ==================================================

if page == "📊 Tableau de bord":

    st.title("📊 Tableau de bord des transactions")

    if df_historique is None:
        # st.warning(
        #    f"Aucune donnée historique trouvée à l'emplacement `{CHEMIN_DONNEES_HISTORIQUES}`. "
        #    "Déposez le fichier à cet endroit pour activer le tableau de bord, "
        #    "ou importez-le ci-dessous."
        #)
        fichier_dashboard = st.file_uploader("Importer un jeu de données  (CSV)", type=["csv"])
        if fichier_dashboard is not None:
            df_historique = pd.read_csv(fichier_dashboard, sep=None, engine="python")
            if "Date" in df_historique.columns:
                df_historique["Date"] = pd.to_datetime(df_historique["Date"], errors="coerce")

    if df_historique is not None:

        # ---------- Filtres ----------
        with st.expander("🔎 Filtres", expanded=False):
            colf1, colf2, colf3 = st.columns(3)
            villes_filtre = colf1.multiselect(
                "Localisation", sorted(df_historique["Localisation"].dropna().unique())
            )
            types_filtre = colf2.multiselect(
                "Type de transaction", sorted(df_historique["Type de transaction"].dropna().unique())
            )
            classes_filtre = colf3.multiselect(
                "Classe (Target)", sorted(df_historique["Target"].dropna().unique())
            )

        df_filtre = df_historique.copy()
        if villes_filtre:
            df_filtre = df_filtre[df_filtre["Localisation"].isin(villes_filtre)]
        if types_filtre:
            df_filtre = df_filtre[df_filtre["Type de transaction"].isin(types_filtre)]
        if classes_filtre:
            df_filtre = df_filtre[df_filtre["Target"].isin(classes_filtre)]

        # ---------- KPI ----------
        total = len(df_filtre)
        nb_fraude = (df_filtre["Target"] == "Fraude").sum()
        taux_fraude = (nb_fraude / total * 100) if total else 0
        montant_moyen = df_filtre["Montant"].mean() if total else 0
        ville_top = (
            df_filtre["Localisation"].value_counts().idxmax()
            if total else "—"
        )

        c1, c2, c3, c4 = st.columns(4)
        carte_kpi(c1, "Transactions", f"{total:,}".replace(",", " "))
        carte_kpi(c2, "Taux de fraude", f"{taux_fraude:.1f} %", couleur=COULEUR_FRAUDE)
        carte_kpi(c3, "Montant moyen", f"{montant_moyen:,.0f} FCFA".replace(",", " "))
        carte_kpi(c4, "Ville la plus active", ville_top)

        st.write("")
        st.write("")

        # ---------- Répartition Target + évolution temporelle ----------
        col_gauche, col_droite = st.columns([1, 1.4])

        with col_gauche:
            st.subheader("Répartition des classes")
            repartition = df_filtre["Target"].value_counts().reset_index()
            repartition.columns = ["Target", "Nombre"]
            fig_donut = px.pie(
                repartition, names="Target", values="Nombre", hole=0.55,
                color="Target", color_discrete_map=PALETTE_TARGET
            )
            fig_donut.update_traces(textinfo="percent+label")
            fig_donut.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=340)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_droite:
            st.subheader("Évolution des transactions dans le temps")
            if df_filtre["Date"].notna().any():
                df_temps = (
                    df_filtre.dropna(subset=["Date"])
                    .assign(Mois=lambda d: d["Date"].dt.to_period("M").dt.to_timestamp())
                    .groupby(["Mois", "Target"]).size().reset_index(name="Nombre")
                )
                fig_ligne = px.line(
                    df_temps, x="Mois", y="Nombre", color="Target",
                    color_discrete_map=PALETTE_TARGET, markers=True
                )
                fig_ligne.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340, legend_title="")
                st.plotly_chart(fig_ligne, use_container_width=True)
            else:
                st.info("Pas de dates exploitables dans les données filtrées.")

        # ---------- Montant par classe + top localisations ----------
        col_g2, col_d2 = st.columns([1, 1.4])

        with col_g2:
            st.subheader("Montant selon la classe")
            fig_box = px.box(
                df_filtre, x="Target", y="Montant", color="Target",
                color_discrete_map=PALETTE_TARGET, points=False
            )
            fig_box.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=340)
            fig_box.update_yaxes(type="log", title="Montant (échelle log)")
            st.plotly_chart(fig_box, use_container_width=True)

        with col_d2:
            st.subheader("Top 10 des localisations")
            top_villes = (
                df_filtre["Localisation"].value_counts().head(10).sort_values().reset_index()
            )
            top_villes.columns = ["Localisation", "Nombre"]
            fig_bar = px.bar(
                top_villes, x="Nombre", y="Localisation", orientation="h",
                color_discrete_sequence=[COULEUR_PRIMAIRE]
            )
            fig_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ---------- Type / statut ----------
        col_g3, col_d3 = st.columns(2)

        with col_g3:
            st.subheader("Taux de fraude par type de transaction")
            taux_type = (
                df_filtre.assign(est_fraude=(df_filtre["Target"] == "Fraude").astype(int))
                .groupby("Type de transaction")["est_fraude"].mean().mul(100).sort_values(ascending=False)
                .reset_index()
            )
            fig_type = px.bar(
                taux_type, x="Type de transaction", y="est_fraude",
                color_discrete_sequence=[COULEUR_FRAUDE]
            )
            fig_type.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, yaxis_title="Taux de fraude (%)")
            st.plotly_chart(fig_type, use_container_width=True)

        with col_d3:
            st.subheader("Statut des opérations par classe")
            df_statut = (
                df_filtre.groupby(["Status operation", "Target"]).size().reset_index(name="Nombre")
            )
            fig_statut = px.bar(
                df_statut, x="Status operation", y="Nombre", color="Target",
                barmode="group", color_discrete_map=PALETTE_TARGET
            )
            fig_statut.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, legend_title="")
            st.plotly_chart(fig_statut, use_container_width=True)


# ==================================================
# 6. PAGE — TRANSACTION UNIQUE
# ==================================================

elif page == "🔍 Transaction unique":

    st.title("🔍 Analyse d'une transaction")
    st.caption("Renseignez les informations d'une transaction pour évaluer son niveau de risque.")

    st.divider()

    colonne1, colonne2 = st.columns(2)

    with colonne1:
        type_transaction = st.selectbox(
            "Type de transaction",
            types_transaction_options,
            key="type_transaction"
        )
        status_operation = st.selectbox(
            "Statut de l'opération",
            status_options,
            key="status_operation"
        )
        montant = st.number_input(
            "Montant de la transaction (FCFA)",
            min_value=0.0, value=50000.0, step=1000.0, key="montant"
        )

    with colonne2:
        index_dakar = localisations.index("Dakar") if "Dakar" in localisations else 0
        localisation = st.selectbox(
            "Localisation", options=localisations, index=index_dakar, key="localisation_transaction"
        )
        date_transaction = st.date_input(
            "Date de la transaction", value=datetime.now().date(), key="date_transaction"
        )
        heure_transaction = st.time_input(
            "Heure de la transaction", value=datetime.now().time(), key="heure_transaction"
        )
        #st.caption(
        #    "ℹ️ L'heure exacte a été encodée comme catégorie pendant l'entraînement : "
        #    "une heure inédite est traitée comme valeur inconnue par le modèle."
        #)

    if st.button("Analyser la transaction", type="primary", use_container_width=True):

        date_complete = datetime.combine(date_transaction, heure_transaction)

        transaction = construire_ligne_modele(
            type_transaction, status_operation, localisation, montant, date_complete
        )

        predictions, probabilites, classes = predire(transaction)
        prediction = predictions[0]
        resultats = dict(zip(classes, probabilites[0]))

        st.divider()
        st.subheader("Résultat de l'analyse")

        col_verdict, col_jauge = st.columns([1, 1])

        with col_verdict:
            if prediction == "Fraude":
                st.error("🚨 Cette transaction est classée comme une **FRAUDE**.")
            elif prediction == "Suspect":
                st.warning("⚠️ Cette transaction est considérée comme **SUSPECTE**.")
            else:
                st.success("✅ Cette transaction est considérée comme **NORMALE**.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Normal", f"{resultats.get('Normal', 0):.1%}")
            col2.metric("Suspect", f"{resultats.get('Suspect', 0):.1%}")
            col3.metric("Fraude", f"{resultats.get('Fraude', 0):.1%}")

            st.write("")
            fig_barres = px.bar(
                x=list(resultats.keys()), y=list(resultats.values()),
                color=list(resultats.keys()), color_discrete_map=PALETTE_TARGET,
                labels={"x": "", "y": "Probabilité"}
            )
            fig_barres.update_layout(showlegend=False, height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%")
            st.plotly_chart(fig_barres, use_container_width=True)

        with col_jauge:
            st.plotly_chart(jauge_risque(resultats.get("Fraude", 0)), use_container_width=True)

            st.markdown("##### Récapitulatif de la transaction")
            st.table(pd.DataFrame({
                "Champ": ["Type", "Statut", "Localisation", "Montant", "Date/Heure"],
                "Valeur": [type_transaction, status_operation, localisation,
                           f"{montant:,.0f} FCFA".replace(",", " "),
                           date_complete.strftime("%d/%m/%Y %H:%M")]
            }).set_index("Champ"))


# ==================================================
# 7. PAGE — FICHIER CSV
# ==================================================

elif page == "📁 Fichier CSV":

    st.title("📁 Analyse d'un fichier CSV")
    st.write(
        "Chargez un fichier contenant plusieurs transactions. "
        "Le fichier doit utiliser la même structure que le jeu de données d'entraînement."
    )

    fichier = st.file_uploader("Sélectionnez un fichier CSV", type=["csv"], key="fichier_csv")

    if fichier is not None:

        try:
            df_original = pd.read_csv(fichier, sep=None, engine="python", encoding="utf-8")

            st.success(f"Fichier chargé : {len(df_original)} transaction(s).")
            st.write("### Aperçu du fichier")
            st.dataframe(df_original.head(10), use_container_width=True)

            colonnes_obligatoires = [
                "Type de transaction", "Status operation", "Localisation", "Date", "Montant"
            ]
            colonnes_absentes = [c for c in colonnes_obligatoires if c not in df_original.columns]

            if colonnes_absentes:
                st.error("Colonnes absentes : " + ", ".join(colonnes_absentes))

            elif st.button("Lancer l'analyse du fichier", type="primary", use_container_width=True):

                X_fichier = preparer_donnees(df_original)

                if X_fichier.isnull().any().any():
                    st.error("Certaines dates ou valeurs du fichier sont incorrectes ou manquantes.")
                else:
                    predictions, probabilites, classes = predire(X_fichier)

                    resultats_csv = df_original.copy()
                    resultats_csv["Prediction IA"] = predictions
                    for position, classe in enumerate(classes):
                        resultats_csv[f"Probabilité {classe}"] = probabilites[:, position]

                    st.divider()
                    st.subheader("Résultats de l'analyse")

                    nombre_normal = (predictions == "Normal").sum()
                    nombre_suspect = (predictions == "Suspect").sum()
                    nombre_fraude = (predictions == "Fraude").sum()

                    c1, c2, c3, c4 = st.columns(4)
                    carte_kpi(c1, "Transactions analysées", f"{len(df_original):,}".replace(",", " "))
                    carte_kpi(c2, "Normales", f"{nombre_normal:,}".replace(",", " "), couleur=COULEUR_NORMAL)
                    carte_kpi(c3, "Suspectes", f"{nombre_suspect:,}".replace(",", " "), couleur=COULEUR_SUSPECT)
                    carte_kpi(c4, "Fraudes détectées", f"{nombre_fraude:,}".replace(",", " "), couleur=COULEUR_FRAUDE)

                    st.write("")

                    col_a, col_b = st.columns([1, 1.5])

                    with col_a:
                        repartition = pd.Series(predictions).value_counts().reset_index()
                        repartition.columns = ["Classe", "Nombre"]
                        fig_pie = px.pie(
                            repartition, names="Classe", values="Nombre", hole=0.55,
                            color="Classe", color_discrete_map=PALETTE_TARGET
                        )
                        fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_b:
                        st.markdown("##### Transactions les plus à risque")
                        top_risque = resultats_csv.sort_values("Probabilité Fraude", ascending=False).head(8)
                        st.dataframe(
                            top_risque[["Localisation", "Type de transaction", "Montant",
                                        "Prediction IA", "Probabilité Fraude"]],
                            use_container_width=True
                        )

                    st.markdown("##### Détail complet des résultats")
                    st.dataframe(resultats_csv, use_container_width=True)

                    fichier_resultat = resultats_csv.to_csv(
                        index=False, sep=";", encoding="utf-8"
                    ).encode("utf-8-sig")

                    st.download_button(
                        "⬇️ Télécharger les résultats",
                        data=fichier_resultat,
                        file_name="resultats_detection_fraude.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as erreur:
            st.error(f"Impossible de traiter le fichier : {erreur}")


# ==================================================
# 8. PAGE — PERFORMANCE DU MODÈLE
# ==================================================

else:

    st.title("🧪 Performance du modèle")
    st.caption("Métriques mesurées sur le jeu de test lors de l'entraînement (Random Forest optimisé).")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    carte_kpi(c1, "Accuracy", f"{METRIQUES_MODELE['accuracy']:.1%}")
    carte_kpi(c2, "Precision", f"{METRIQUES_MODELE['precision']:.1%}")
    carte_kpi(c3, "Recall", f"{METRIQUES_MODELE['recall']:.1%}")
    carte_kpi(c4, "F1-score", f"{METRIQUES_MODELE['f1']:.1%}")

    st.write("")
    st.write("")

    col_gauche, col_droite = st.columns([1.2, 1])

    with col_gauche:
        st.subheader("Matrice de confusion")
        mc = METRIQUES_MODELE["matrice_confusion"]
        fig_mc = px.imshow(
            mc["valeurs"], x=mc["labels"], y=mc["labels"],
            text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Prédit", y="Réel", color="Nombre")
        )
        fig_mc.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_droite:
        st.subheader("Précision / Rappel / F1 par classe")
        df_par_classe = pd.DataFrame(METRIQUES_MODELE["par_classe"]).T.reset_index()
        df_par_classe.columns = ["Classe", "Precision", "Recall", "F1", "Support"]
        df_long = df_par_classe.melt(
            id_vars=["Classe", "Support"], value_vars=["Precision", "Recall", "F1"],
            var_name="Métrique", value_name="Valeur"
        )
        fig_classe = px.bar(
            df_long, x="Classe", y="Valeur", color="Métrique", barmode="group",
            color_discrete_sequence=[COULEUR_PRIMAIRE, "#4A6FA5", COULEUR_FRAUDE]
        )
        fig_classe.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig_classe, use_container_width=True)

    st.write("")
    st.markdown("##### Détail par classe")
    st.dataframe(
        df_par_classe.style.format({"Precision": "{:.0%}", "Recall": "{:.0%}", "F1": "{:.0%}"}),
        use_container_width=True
    )

    # ---------- Importance des variables ----------
    st.write("")
    st.subheader("Importance des variables")
    try:
        importances = pd.DataFrame({
            "Variable": feature_columns,
            "Importance": modele.feature_importances_
        }).sort_values("Importance", ascending=True).tail(12)

        fig_imp = px.bar(
            importances, x="Importance", y="Variable", orientation="h",
            color_discrete_sequence=[COULEUR_PRIMAIRE]
        )
        fig_imp.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception:
        st.info("Importance des variables non disponible pour ce modèle.")

    st.caption(
        "ℹ️ Ces métriques sont figées au moment de l'entraînement. "
        "Pour les recalculer automatiquement, sauvegardez-les dans un fichier "
        "(ex. `model/metrics.json`) lors de l'entraînement et chargez-les ici."
    )