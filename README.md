# Open-Data-Projet

# 📉 Prédiction de Désabonnement (Churn Prediction Dashboard)

## 📌 Description

Le désabonnement (*churn*) est un enjeu majeur pour les entreprises, notamment dans les télécommunications, la banque ou les services numériques.
Ce projet vise à construire un **tableau de bord interactif** permettant :

* d’analyser les comportements clients,
* de prédire la probabilité de désabonnement,
* et de visualiser les résultats sous forme de graphiques et cartes interactives.

---

## 📊 Données utilisées

### 1. **Dataset Public : IBM Telco Customer Churn** *(recommandé)*

* ~7 000 clients de télécommunications
* **Variables disponibles :**

  * Informations démographiques : genre, âge, situation familiale
  * Services souscrits : téléphone, internet (DSL, fibre), sécurité en ligne…
  * Informations de compte : ancienneté, contrat, facturation, frais mensuels
  * **Variable cible :** `Churn` (binaire : Oui / Non)

👉 Disponible sur Kaggle : *IBM Telco Customer Churn Dataset*

### 2. **Autres datasets**

* **Banque / cartes de crédit** : informations clients, produits détenus, score de crédit, géographie.
* Pertinence géographique : permet d’intégrer une **carte interactive** montrant les zones à risque.

### 3. **Données réelles ou simulées** *(optionnel)*

* Connexion à une base de données SQL/NoSQL
* API interne pour récupérer des métriques comportementales en temps réel

---

## 🛠️ Technologies clés

* **Python** (Pandas, Scikit-learn, NumPy)
* **Visualisation** : Streamlit ou Dash pour le tableau de bord interactif
* **Machine Learning** : modèles de classification (Logistic Regression, Random Forest, XGBoost…)
* **Stockage** : CSV / Base de données (optionnel pour un cas réel)

---

## 🚀 Fonctionnalités prévues

* Import et nettoyage des données
* Entraînement de modèles de prédiction du churn
* Évaluation des performances (précision, rappel, F1-score, AUC)
* Visualisations interactives :

  * Distribution des clients (par âge, contrat, services souscrits…)
  * Importance des variables (features les plus corrélées au churn)
  * Taux de churn par région (via carte interactive)
* Prédiction en direct : tester un profil client et obtenir la probabilité de churn

---

## 📦 Livrables

* Script de préparation et modélisation des données
* Tableau de bord interactif (Streamlit/Dash)
* Documentation pour reproduire les résultats


