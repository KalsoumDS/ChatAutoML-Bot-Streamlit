# 📥 Comment Obtenir les Datasets

## 🚀 Méthode 1 : Script Automatique (Recommandé)

Exécutez simplement cette commande dans votre terminal :

```bash
cd "/Users/macbook/Desktop/IA Avancées Project"
python datasets/download_datasets.py
```

**Ce script va :**
- ✅ Télécharger les datasets réels (Titanic, Iris, Wine Quality, etc.)
- ✅ Créer automatiquement 8 datasets d'exemple variés
- ✅ Tout sauvegarder dans le dossier `datasets/`

## 📊 Datasets Disponibles

### ✅ Datasets Générés (Toujours Disponibles)

Ces datasets sont créés automatiquement et ne nécessitent pas de connexion internet :

1. **sample_binary_classification.csv** - Classification binaire équilibrée
2. **sample_imbalanced_binary.csv** - Classification binaire DÉSÉQUILIBRÉE
3. **sample_multiclass_classification.csv** - Classification multiclasse (3 classes)
4. **sample_regression.csv** - Régression simple
5. **sample_with_missing_values.csv** - Avec valeurs manquantes
6. **sample_mixed_features.csv** - Features numériques + catégorielles
7. **sample_regression_outliers.csv** - Régression avec outliers
8. **sample_small.csv** - Petit dataset (50 échantillons) pour tests rapides

### 📥 Datasets Réels (Nécessitent Internet)

Si le téléchargement automatique échoue (problème SSL), vous pouvez les télécharger manuellement :

#### 1. **Iris** ✅ (Déjà téléchargé)
- **Fichier** : `datasets/iris.csv`
- **Colonne cible** : `species` ou `target`
- **Type** : Classification multiclasse

#### 2. **Titanic**
- **Source** : https://www.kaggle.com/c/titanic/data
- **Ou** : https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- **Colonne cible** : `Survived`
- **Type** : Classification binaire

#### 3. **California Housing**
- **Source** : sklearn (via Python)
- **Commande** :
  ```python
  from sklearn.datasets import fetch_california_housing
  import pandas as pd
  housing = fetch_california_housing()
  df = pd.DataFrame(housing.data, columns=housing.feature_names)
  df['MedHouseVal'] = housing.target
  df.to_csv('datasets/california_housing.csv', index=False)
  ```
- **Colonne cible** : `MedHouseVal`
- **Type** : Régression

#### 4. **Wine Quality**
- **Source** : https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
- **Fichiers** : `winequality-red.csv` et `winequality-white.csv`
- **Colonne cible** : `quality`
- **Type** : Classification ou Régression

#### 5. **Bank Marketing**
- **Source** : https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Colonne cible** : `y`
- **Type** : Classification binaire déséquilibrée

#### 6. **Student Performance**
- **Source** : https://archive.ics.uci.edu/dataset/320/student+performance
- **Colonne cible** : Variable selon le fichier
- **Type** : Classification ou Régression

## 🎯 Utilisation dans ChatAutoML-Bot

Une fois les datasets dans le dossier `datasets/` :

1. **Lancez l'application** :
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Dans l'interface** :
   - Cliquez sur "📁 Charger un Dataset" dans le menu latéral
   - Sélectionnez un fichier CSV depuis `datasets/`
   - Choisissez la colonne cible
   - Lancez AutoML !

## 💡 Recommandations

### Pour débuter rapidement :
- ✅ **sample_small.csv** - Très rapide (50 échantillons)
- ✅ **iris.csv** - Dataset classique, bien équilibré
- ✅ **sample_binary_classification.csv** - Classification simple

### Pour tester toutes les fonctionnalités :
- ✅ **sample_imbalanced_binary.csv** - Teste le déséquilibre
- ✅ **sample_with_missing_values.csv** - Teste l'imputation
- ✅ **sample_mixed_features.csv** - Teste le preprocessing complet

## 🔧 Résolution de Problèmes

### Problème SSL (Certificat)
Si vous voyez des erreurs SSL lors du téléchargement :

**Solution 1** : Téléchargez manuellement depuis les liens fournis ci-dessus

**Solution 2** : Désactivez temporairement la vérification SSL (non recommandé pour la production) :
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### Datasets Manquants
Les datasets d'exemple sont **toujours créés** même si les téléchargements échouent. Vous avez donc au minimum 8 datasets pour tester l'application !

## 📝 Note

Les datasets générés utilisent `random_state=42` pour garantir la reproductibilité. Vous obtiendrez toujours les mêmes résultats.

---

**Bon entraînement ! 🚀**


