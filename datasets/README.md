# Datasets de Démonstration

Ce dossier contient les jeux de données de référence pour tester ChatAutoML-Bot.

## 📥 Téléchargement automatique

Pour télécharger automatiquement tous les datasets :

```bash
python datasets/download_datasets.py
```

## 📊 Jeux de Données Disponibles

### Classification

1. **Titanic** (`titanic.csv`) - Classification binaire
   - Source: Kaggle - https://www.kaggle.com/c/titanic/data
   - Tâche: Prédire la survie des passagers
   - Colonne cible: `Survived`
   - ~890 lignes

2. **Iris** (`iris.csv`) - Classification multiclasse
   - Source: sklearn / UCI - https://archive.ics.uci.edu/dataset/53/iris
   - Tâche: Classifier les espèces d'iris
   - Colonne cible: `species` ou `target`
   - 150 lignes

3. **Bank Marketing** (`bank_marketing.csv`) - Classification binaire déséquilibrée
   - Source: UCI - https://archive.ics.uci.edu/dataset/222/bank+marketing
   - Tâche: Prédire si un client souscrira à un dépôt à terme
   - Colonne cible: `y`
   - ~45,000 lignes

### Régression

1. **California Housing** (`california_housing.csv`) - Régression
   - Source: sklearn.datasets.fetch_california_housing
   - Tâche: Prédire le prix médian des maisons
   - Colonne cible: `target` ou `MedHouseVal`
   - ~20,000 lignes

2. **Wine Quality** (`wine_quality_red.csv`, `wine_quality_white.csv`) - Régression ou Classification
   - Source: UCI - https://archive.ics.uci.edu/dataset/186/wine+quality
   - Tâche: Prédire la qualité du vin (0-10)
   - Colonne cible: `quality`
   - ~1,600 lignes (rouge) et ~5,000 lignes (blanc)

3. **Student Performance** (`student_performance_*.csv`) - Régression ou Classification
   - Source: UCI - https://archive.ics.uci.edu/dataset/320/student+performance
   - Tâche: Prédire les performances des étudiants
   - Colonne cible: `G3` (note finale)
   - ~395 lignes

### Datasets d'exemple

- **sample_binary_classification.csv** - Dataset simple pour tests de classification
- **sample_regression.csv** - Dataset simple pour tests de régression

## 💻 Utilisation

1. **Télécharger les datasets** :
   ```bash
   cd "/Users/macbook/Desktop/IA Avancées Project"
   python datasets/download_datasets.py
   ```

2. **Dans l'interface Streamlit** :
   - Cliquez sur "Charger un Dataset" dans la sidebar
   - Naviguez vers le dossier `datasets/`
   - Sélectionnez un fichier CSV
   - Choisissez la colonne cible appropriée

## 📝 Notes

- Certains datasets peuvent nécessiter un téléchargement manuel si les liens automatiques ne fonctionnent pas
- Les datasets sont téléchargés au format CSV pour faciliter l'utilisation
- Les fichiers sont sauvegardés dans ce dossier (`datasets/`)

## 🔗 Liens de téléchargement manuel

Si le script automatique ne fonctionne pas, vous pouvez télécharger manuellement :

- **Titanic** : https://www.kaggle.com/c/titanic/data (nécessite un compte Kaggle)
- **Iris** : Disponible via sklearn ou UCI
- **Bank Marketing** : https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Wine Quality** : https://archive.ics.uci.edu/dataset/186/wine+quality
- **California Housing** : Disponible via sklearn
- **Student Performance** : https://archive.ics.uci.edu/dataset/320/student+performance
