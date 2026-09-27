# HousePrice AI

Projet de Machine Learning pour estimer le prix de vente d'un logement à partir de ses caractéristiques.

## Description

Cette application combine un pipeline de préparation des données, plusieurs modèles de régression et un tableau de bord Streamlit pour prédire le prix d'un bien immobilier.

Le projet utilise des données immobilières de type Kaggle-style, applique des transformations sur les variables, entraîne des modèles supervisés et expose une interface simple pour faire des prédictions interactives.

## Objectif

- préparer les données de logement,
- comparer plusieurs algorithmes de régression,
- enregistrer le modèle final,
- proposer une estimation de prix via une interface web.

## Structure du projet

```text
HousePrice AI/
├── dashboard/
│   └── app.py                  # Interface Streamlit de prédiction
├── data/
│   ├── House_Prices.csv       # Jeu de données brut
│   ├── House_PricesV1.csv      # Données nettoyées / préparées
│   ├── saleprice_ronds.csv     # Données filtrées / arrondies
│   ├── dataFeature.csv         # Jeu enrichi avec variables dérivées
│   └── ...
├── models/
│   ├── modele_lineaire.py      # Script d'entraînement des modèles
│   └── house_price_model.pkl   # Modèle sauvegardé (généré après entraînement)
├── notebooks/
│   └── exploration_data.py     # Analyse exploratoire et création des features
├── docker-compose.yml          # Déploiement multi-services Docker
├── Dockerfile                  # Image Python du projet
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation du projet
└── tests/                      # Dossier de tests (si ajouté plus tard)
```

## Stack technique

- Python 3.11
- pandas
- scikit-learn
- Streamlit
- matplotlib
- seaborn
- Docker / Docker Compose

## Installation

### 1. Cloner le projet

```bash
git clone <url-du-projet>
cd "HousePrice AI"
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Entraînement du modèle

Le script principal d'entraînement est situé dans :

```bash
python models/modele_lineaire.py
```

Ce script :

- charge les données,
- effectue une préparation et un nettoyage de base,
- construit plusieurs modèles,
- compare leurs performances,
- sauvegarde le meilleur modèle dans `models/house_price_model.pkl`.

## Lancement de l'application Streamlit

Depuis la racine du projet :

```bash
streamlit run dashboard/app.py
```

L'application sera accessible sur :

- http://localhost:8501

## Lancement via Docker

Le projet contient un fichier `docker-compose.yml` configuré pour lancer la partie analyse et l'interface de prédiction.

```bash
docker compose up --build
```

### Services disponibles

- `streamlit` : interface de prédiction sur le port `8501`
- `app` : service backend / traitement associé

## Utilisation

1. Ouvrir le dashboard Streamlit.
2. Saisir les caractéristiques du logement.
3. Cliquer sur le bouton de prédiction.
4. Lire la valeur estimée du prix.

## Données

Les fichiers du dossier `data/` représentent les différentes étapes de préparation des données :

- `House_Prices.csv` : jeu brut,
- `House_PricesV1.csv` : données nettoyées,
- `saleprice_ronds.csv` : version filtrée pour l'analyse,
- `dataFeature.csv` : variables enrichies utiles à la modélisation.

## Notes

- Le modèle est chargé par le fichier `dashboard/app.py` depuis le dossier `models/`.
- Le dashboard utilise les colonnes attendues par le modèle, donc il est important de garder la structure de données cohérente après entraînement.
- Si le fichier `house_price_model.pkl` n'existe pas, il faut d'abord lancer le script d'entraînement.

## Prochaines améliorations possibles

- ajouter un meilleur prétraitement des variables catégorielles,
- comparer davantage de modèles,
- ajouter des métriques et graphiques de performance,
- mettre en place des tests automatiques,
- déployer l'application dans un environnement cloud ou conteneurisé.

## Auteur

Projet personnel / étude de cas sur la prédiction immobilière par Machine Learning.