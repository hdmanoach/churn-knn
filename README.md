# 📝 Rapport de Projet : Prédiction du Churn (k-NN)

---
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hdmanoach/churn-knn/blob/main/notebooks/churn+.ipynb)

## 1. Introduction

Le but de ce projet est de construire un modèle de **classification supervisée** pour prédire si un client va quitter (churn) un service télécom.

Nous utilisons un dataset contenant **7043 clients** avec plusieurs caractéristiques : genre, durée d’abonnement (tenure), type de contrat, facturation, services internet, etc.

L’objectif est de prédire la variable cible :

* **Churn = Yes** → le client quitte
* **Churn = No** → le client reste

---
## 🧠 Méthodologie
1. Chargement des données
2. Nettoyage et traitement
3. Encodage des variables catégorielles (One-Hot)
4. Normalisation (StandardScaler)
5. Split train/test (80/20)
6. Entraînement k-NN
7. Optimisation de k avec GridSearchCV
8. Évaluation (F1-score, confusion matrix)

## 2. Description des données

Le dataset contient **21 colonnes** :

* `customerID` : identifiant du client
* `Churn` : variable cible (Yes/No)
* Variables catégorielles :

  * Contract
  * InternetService
  * PaymentMethod
  * OnlineSecurity
  * etc.
* Variables numériques :

  * tenure
  * MonthlyCharges
  * TotalCharges

### Distribution de la variable cible

| Classe | Nombre |
| ------ | ------ |
| No     | 5174   |
| Yes    | 1869   |

Le dataset est **déséquilibré** : il y a plus de clients qui restent que de clients qui partent.

---

## 3. Préparation des données

### 3.1 Nettoyage

* Vérification des valeurs manquantes
* La colonne `TotalCharges` contenait 11 valeurs manquantes
* Après traitement, aucune valeur manquante ne reste

---

### 3.2 Séparation des variables

* Suppression de `customerID` (non utile pour la prédiction)
* Définition :

```python
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})
```

* `X` : variables explicatives
* `y` : variable cible (0 = No, 1 = Yes)

---

### 3.3 Encodage des variables catégorielles

Les variables catégorielles ont été transformées en variables numériques avec :

```python
pd.get_dummies()
```

Cela permet au modèle k-NN de travailler avec des données numériques.

---

### 3.4 Normalisation

Les données ont été normalisées avec :

```python
StandardScaler()
```

La normalisation est essentielle pour k-NN car ce modèle est basé sur la **distance** entre les points.

---

## 4. Séparation Train / Test

Le dataset a été divisé en :

* **80% données d'entraînement**
* **20% données de test**

Avec :

```python
train_test_split(..., stratify=y)
```

Cela permet de garder la même proportion de classes dans train et test.

---

## 5. Modèle k-NN

### 5.1 Modèle initial (k = 5)

Résultats :

* Accuracy : **0.76**
* F1-score (churn) : **0.54**

Matrice de confusion :

```
[[861, 172],
 [172, 202]]
```

Interprétation :

* 861 clients correctement prédits comme restant
* 202 clients correctement prédits comme churn
* 172 faux positifs
* 172 faux négatifs

---

## 6. Optimisation du modèle

Une recherche du meilleur `k` a été effectuée avec :

```python
GridSearchCV
```

Test de k de 1 à 30 en optimisant le **F1-score**.

### Résultats :

* **Meilleur k : 25**
* **Meilleur F1-score : 0.5848**

---

## 7. Modèle final (k = 25)

Matrice de confusion :

```
[[884, 149],
 [160, 214]]
```

### Performances

| Classe | Precision | Recall | F1-score | Support |
| ------ | --------- | ------ | -------- | ------- |
| No     | 0.85      | 0.86   | 0.85     | 1033    |
| Yes    | 0.59      | 0.57   | 0.58     | 374     |

* Accuracy : **0.78**
* F1-score (churn) : **0.58**

---

## 8. Analyse des résultats

* Le modèle est **très bon pour prédire les clients qui restent**.
* Il est **moins performant pour détecter les churns**, ce qui est logique car les classes sont déséquilibrées.
* Les faux négatifs (clients qui partent mais non détectés) ont diminué de **172 à 160** après optimisation.

L’optimisation du paramètre `k` a donc amélioré les performances.

---

## 9. Conclusion

Le modèle k-NN permet une prédiction correcte du churn avec :

* Accuracy : 78%
* F1-score churn : 0.58

L’optimisation améliore les performances mais des améliorations restent possibles.

---

## 10. Améliorations possibles

* Tester d’autres modèles :

  * Logistic Regression
  * Random Forest
  * XGBoost
* Utiliser SMOTE pour équilibrer les classes
* Sélection des meilleures variables (feature selection)
* Ajuster le seuil de décision pour améliorer le recall
* Déployer le modèle via une application web (Flask / FastAPI)

---

## 11. Technologies utilisées

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Google Colab
## 📥 Fichiers

- 📁 data/ : dataset utilisé
- 📁 notebooks/ : analyses et modèles pas à pas

# 📌 Résumé rapide

Ce projet montre la mise en place complète d’un pipeline de machine learning :

* Nettoyage des données
* Encodage
* Normalisation
* Entraînement
* Optimisation
* Évaluation

Le modèle k-NN optimisé (k=25) atteint un F1-score de **0.58** sur la classe churn.
