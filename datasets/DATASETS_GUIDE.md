# Guide d'utilisation des Datasets

## 📊 Datasets disponibles et colonnes cibles

### Classification

#### 1. **Titanic** (`titanic.csv`)
- **Type** : Classification binaire
- **Colonne cible** : `Survived` (0 ou 1)
- **Description** : Prédire si un passager a survécu au naufrage
- **Taille** : 891 lignes, 12 colonnes
- **Note** : Dataset classique pour débuter en ML

#### 2. **Iris** (`iris.csv`)
- **Type** : Classification multiclasse
- **Colonne cible** : `species` ou `target` (setosa, versicolor, virginica)
- **Description** : Classifier les espèces d'iris selon leurs caractéristiques
- **Taille** : 150 lignes, 6 colonnes
- **Note** : Dataset très simple, parfait pour les tests

#### 3. **Bank Marketing** (`bank_marketing.csv`)
- **Type** : Classification binaire (déséquilibrée)
- **Colonne cible** : `y` (yes/no)
- **Description** : Prédire si un client souscrira à un dépôt à terme
- **Taille** : 4,521 lignes, 17 colonnes
- **Note** : Dataset déséquilibré, bon pour tester SMOTE

### Régression

#### 4. **California Housing** (`california_housing.csv`)
- **Type** : Régression
- **Colonne cible** : `target` ou `MedHouseVal`
- **Description** : Prédire le prix médian des maisons en Californie
- **Taille** : 20,640 lignes, 10 colonnes
- **Note** : Dataset de régression classique

#### 5. **Wine Quality - Rouge** (`wine_quality_red.csv`)
- **Type** : Régression ou Classification
- **Colonne cible** : `quality` (0-10)
- **Description** : Prédire la qualité du vin rouge
- **Taille** : 1,599 lignes, 12 colonnes
- **Note** : Peut être utilisé en classification (qualité basse/moyenne/haute)

#### 6. **Wine Quality - Blanc** (`wine_quality_white.csv`)
- **Type** : Régression ou Classification
- **Colonne cible** : `quality` (0-10)
- **Description** : Prédire la qualité du vin blanc
- **Taille** : 4,898 lignes, 12 colonnes
- **Note** : Version blanche, plus de données que la version rouge

#### 7. **Student Performance - Math** (`student_performance_student-mat.csv`)
- **Type** : Régression ou Classification
- **Colonne cible** : `G3` (note finale de 0 à 20)
- **Description** : Prédire les performances en mathématiques
- **Taille** : ~395 lignes
- **Note** : Peut être utilisé en classification (note basse/moyenne/haute)

#### 8. **Student Performance - Portuguese** (`student_performance_student-por.csv`)
- **Type** : Régression ou Classification
- **Colonne cible** : `G3` (note finale de 0 à 20)
- **Description** : Prédire les performances en portugais
- **Taille** : ~649 lignes
- **Note** : Version portugais, plus de données que math

### Datasets d'exemple

#### 9. **Sample Binary Classification** (`sample_binary_classification.csv`)
- **Type** : Classification binaire
- **Colonne cible** : `target` (class_A, class_B)
- **Description** : Dataset simple généré pour tests
- **Taille** : 400 lignes, 3 colonnes
- **Note** : Parfait pour tester rapidement

#### 10. **Sample Regression** (`sample_regression.csv`)
- **Type** : Régression
- **Colonne cible** : `target`
- **Description** : Dataset simple généré pour tests
- **Taille** : 200 lignes, 4 colonnes
- **Note** : Parfait pour tester rapidement

## 🚀 Utilisation dans ChatAutoML-Bot

1. **Lancer l'application** :
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Charger un dataset** :
   - Cliquez sur "📁 Charger un Dataset" dans la sidebar
   - Naviguez vers le dossier `datasets/`
   - Sélectionnez un fichier CSV

3. **Sélectionner la colonne cible** :
   - Utilisez le menu déroulant "🎯 Colonne Cible"
   - Choisissez la colonne appropriée selon le guide ci-dessus

4. **Lancer AutoML** :
   - Le système détectera automatiquement le type de tâche
   - Cliquez sur "🚀 Lancer AutoML" pour démarrer

## 💡 Conseils

- **Pour débuter** : Utilisez `iris.csv` ou `sample_binary_classification.csv` (petits datasets, rapides)
- **Pour tester le déséquilibre** : Utilisez `bank_marketing.csv`
- **Pour la régression** : Utilisez `california_housing.csv` ou `wine_quality_red.csv`
- **Pour des données réelles** : Utilisez `titanic.csv` (très populaire en ML)

## 📝 Notes importantes

- Certains datasets peuvent avoir des valeurs manquantes (ex: Titanic)
- Les datasets d'étudiants ont plusieurs colonnes de notes (G1, G2, G3) - utilisez G3 comme cible
- Wine Quality peut être utilisé en classification en créant des catégories (ex: qualité < 6 = "basse", etc.)


