# 📊 Liste des Datasets Disponibles

Ce document liste tous les datasets disponibles pour tester ChatAutoML-Bot.

## 🚀 Comment utiliser

1. **Télécharger les datasets** :
   ```bash
   python datasets/download_datasets.py
   ```

2. **Charger dans l'application** :
   - Utilisez le menu latéral "📁 Charger un Dataset"
   - Sélectionnez un fichier CSV depuis le dossier `datasets/`

---

## 📋 Datasets Réels (Téléchargés)

### 1. **Titanic** (`titanic.csv`)
- **Type** : Classification binaire
- **Colonne cible** : `Survived`
- **Description** : Prédire la survie des passagers du Titanic
- **Lignes** : ~891
- **Caractéristiques** : Valeurs manquantes, features mixtes (numériques + catégorielles)
- **Source** : Kaggle

### 2. **Iris** (`iris.csv`)
- **Type** : Classification multiclasse (3 classes)
- **Colonne cible** : `species` ou `target`
- **Description** : Classification des fleurs d'iris en 3 espèces
- **Lignes** : 150
- **Caractéristiques** : Dataset équilibré, toutes features numériques
- **Source** : UCI / sklearn

### 3. **California Housing** (`california_housing.csv`)
- **Type** : Régression
- **Colonne cible** : `MedHouseVal` ou `target`
- **Description** : Prédire le prix médian des maisons en Californie
- **Lignes** : ~20,640
- **Caractéristiques** : Features numériques, valeurs continues
- **Source** : sklearn

### 4. **Wine Quality** (`wine_quality_red.csv`, `wine_quality_white.csv`)
- **Type** : Classification ou Régression
- **Colonne cible** : `quality`
- **Description** : Évaluer la qualité du vin (rouge ou blanc)
- **Lignes** : ~1,599 (rouge), ~4,898 (blanc)
- **Caractéristiques** : Features numériques, peut être utilisé pour classification ou régression
- **Source** : UCI

### 5. **Bank Marketing** (`bank_marketing.csv`)
- **Type** : Classification binaire DÉSÉQUILIBRÉE
- **Colonne cible** : `y` (ou `deposit`)
- **Description** : Prédire si un client souscrira à un dépôt à terme
- **Lignes** : ~4,521
- **Caractéristiques** : Fort déséquilibre des classes, features mixtes
- **Source** : UCI

### 6. **Student Performance** (`student_performance_*.csv`)
- **Type** : Classification ou Régression
- **Colonne cible** : Variable selon le fichier (ex: `G3`)
- **Description** : Prédire les performances scolaires des étudiants
- **Lignes** : Variable
- **Caractéristiques** : Features mixtes (numériques + catégorielles)
- **Source** : UCI

---

## 🧪 Datasets d'Exemple (Générés)

Ces datasets sont créés automatiquement pour tester différentes fonctionnalités :

### 7. **Classification Binaire Équilibrée** (`sample_binary_classification.csv`)
- **Type** : Classification binaire
- **Colonne cible** : `target`
- **Description** : Dataset simple avec 2 classes équilibrées
- **Lignes** : 400
- **Caractéristiques** : 2 features numériques, classes bien séparées
- **Usage** : Test rapide de classification

### 8. **Classification Binaire DÉSÉQUILIBRÉE** (`sample_imbalanced_binary.csv`)
- **Type** : Classification binaire DÉSÉQUILIBRÉE
- **Colonne cible** : `target`
- **Description** : Dataset avec déséquilibre important (800 vs 50)
- **Lignes** : 850
- **Caractéristiques** : 3 features numériques, ratio ~16:1
- **Usage** : Tester la gestion du déséquilibre (SMOTE, oversampling)

### 9. **Classification Multiclasse** (`sample_multiclass_classification.csv`)
- **Type** : Classification multiclasse (3 classes)
- **Colonne cible** : `target`
- **Description** : Dataset avec 3 classes équilibrées
- **Lignes** : 450
- **Caractéristiques** : 2 features numériques, 3 classes
- **Usage** : Tester la classification multiclasse

### 10. **Régression Simple** (`sample_regression.csv`)
- **Type** : Régression
- **Colonne cible** : `target`
- **Description** : Dataset de régression linéaire simple
- **Lignes** : 200
- **Caractéristiques** : 3 features numériques, relation linéaire
- **Usage** : Test rapide de régression

### 11. **Avec Valeurs Manquantes** (`sample_with_missing_values.csv`)
- **Type** : Régression
- **Colonne cible** : `target`
- **Description** : Dataset avec ~10% de valeurs manquantes
- **Lignes** : 300
- **Caractéristiques** : 4 features numériques, valeurs manquantes dans feature_2 et feature_3
- **Usage** : Tester l'imputation des valeurs manquantes

### 12. **Features Mixtes** (`sample_mixed_features.csv`)
- **Type** : Régression
- **Colonne cible** : `target`
- **Description** : Dataset avec features numériques ET catégorielles
- **Lignes** : 250
- **Caractéristiques** : 2 features numériques + 1 feature catégorielle
- **Usage** : Tester le preprocessing (encodage OneHot)

### 13. **Régression avec Outliers** (`sample_regression_outliers.csv`)
- **Type** : Régression
- **Colonne cible** : `target`
- **Description** : Dataset avec quelques valeurs aberrantes
- **Lignes** : 200
- **Caractéristiques** : 3 features numériques, 10 outliers
- **Usage** : Tester la robustesse des modèles aux outliers

### 14. **Dataset Petit** (`sample_small.csv`)
- **Type** : Classification binaire
- **Colonne cible** : `target`
- **Description** : Dataset très petit pour tests rapides
- **Lignes** : 50
- **Caractéristiques** : 2 features numériques, 2 classes
- **Usage** : Tests rapides et démonstrations

---

## 📊 Résumé par Type

### Classification
- **Binaire équilibrée** : Titanic, sample_binary_classification, sample_small
- **Binaire déséquilibrée** : Bank Marketing, sample_imbalanced_binary
- **Multiclasse** : Iris, sample_multiclass_classification

### Régression
- **Simple** : California Housing, sample_regression
- **Avec outliers** : sample_regression_outliers
- **Avec valeurs manquantes** : sample_with_missing_values
- **Features mixtes** : sample_mixed_features

### Caractéristiques Spéciales
- **Valeurs manquantes** : Titanic, sample_with_missing_values
- **Déséquilibre** : Bank Marketing, sample_imbalanced_binary
- **Features mixtes** : Titanic, Bank Marketing, Student Performance, sample_mixed_features
- **Outliers** : sample_regression_outliers

---

## 💡 Recommandations d'Usage

### Pour débuter
1. **Iris** : Dataset simple, équilibré, classification multiclasse
2. **sample_binary_classification** : Test rapide de classification binaire
3. **sample_regression** : Test rapide de régression

### Pour tester le preprocessing
1. **Titanic** : Valeurs manquantes + features mixtes
2. **sample_with_missing_values** : Beaucoup de valeurs manquantes
3. **sample_mixed_features** : Features numériques et catégorielles

### Pour tester le déséquilibre
1. **Bank Marketing** : Déséquilibre réel
2. **sample_imbalanced_binary** : Déséquilibre extrême (16:1)

### Pour tests rapides
1. **sample_small** : 50 échantillons seulement
2. **Iris** : 150 échantillons

### Pour tests complets
1. **California Housing** : Grande taille, régression
2. **Wine Quality** : Plusieurs milliers d'échantillons

---

## 🔧 Notes Techniques

- Tous les datasets sont au format **CSV**
- La colonne cible peut avoir différents noms selon le dataset
- Les datasets générés utilisent `random_state=42` pour la reproductibilité
- Les datasets réels peuvent nécessiter un téléchargement internet

---

**Bon entraînement ! 🚀**


