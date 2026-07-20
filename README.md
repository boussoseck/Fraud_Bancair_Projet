#  Détection de Fraude Bancaire avec Machine Learning

##  Description

Ce projet est une application web développée avec **Streamlit** permettant de détecter automatiquement les transactions bancaires frauduleuses à l'aide d'un modèle de **Machine Learning**.

L'utilisateur peut :

- Analyser une transaction bancaire en renseignant ses caractéristiques.
- Importer un fichier CSV contenant plusieurs transactions.
- Obtenir une prédiction pour chaque transaction.
- Visualiser les probabilités associées à chaque classe.
- Télécharger les résultats de l'analyse.

Le modèle classe chaque transaction dans l'une des catégories suivantes :

-  Normal
-  Suspect
-  Fraude

---

#  Objectifs

Ce projet vise à :

- Développer un système intelligent de détection de fraude bancaire.
- Automatiser l'analyse des transactions.
- Fournir une interface simple et intuitive grâce à Streamlit.
- Mettre en pratique les techniques de Machine Learning.

---

#  Technologies utilisées

- Python 3
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

#  Modèle de Machine Learning

Le modèle utilisé est un :

**Random Forest Classifier**

Ce modèle est particulièrement adapté aux problèmes de classification grâce à :

- une bonne précision ;
- une bonne résistance au surapprentissage ;
- une excellente gestion des variables numériques et catégorielles.

Le modèle entraîné est sauvegardé dans :

```text
model/fraud_model.pkl
```

---

#  Structure du projet

```text
PROJET_FRAUDE_DETECTION
│
├── data/
│   └── Bank_transaction_scenario1.csv
│
├── model/
│   ├── fraud_model.pkl
│   └── train_model.py
│
├── venv/
│
├── app.py
├── README.md
├── requirements.txt
├── resultats_detection_fraude.csv
└── .gitignore
```

---

#  Fonctionnalités

##  Analyse d'une transaction

L'utilisateur renseigne les informations d'une transaction.

Le système affiche :

- la classe prédite ;
- les probabilités de chaque classe.

---

##  Analyse d'un fichier CSV

L'utilisateur importe un fichier CSV.

L'application :

- analyse automatiquement toutes les transactions ;
- ajoute une colonne contenant les prédictions ;
- permet de télécharger les résultats.

---

##  Téléchargement des résultats

Les résultats peuvent être exportés au format CSV après l'analyse.

---

#  Installation

## 1. Cloner le dépôt

```bash
git clone https://github.com/boussoseck/Projet_Fraude_Detection.git
```

---

## 2. Accéder au projet

```bash
cd Projet_Fraude_Detection
```

---

## 3. Créer un environnement virtuel

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 5. Entraîner le modèle

```bash
python model/train_model.py
```

---

## 6. Lancer l'application

```bash
streamlit run app.py
```

Puis ouvrir dans votre navigateur :

```
http://localhost:8501
```

---

#  Jeu de données

Le projet utilise un jeu de données de transactions bancaires contenant plusieurs informations telles que :

- Montant de la transaction
- Type de transaction
- Canal utilisé
- Localisation
- Heure
- Statut
- Autres caractéristiques permettant de détecter les comportements frauduleux.

---

#  Résultats

L'application permet de :

- détecter automatiquement les transactions frauduleuses ;
- analyser plusieurs transactions simultanément ;
- afficher les probabilités associées à chaque classe ;
- exporter les résultats.


---

# Auteur

**Bousso Seck**

Master 1 Intelligence Artificielle (Big Data & IA)

Projet académique réalisé dans le cadre du cours de **Machine Learning**.

---

# Licence

Ce projet est destiné à un usage pédagogique et académique.