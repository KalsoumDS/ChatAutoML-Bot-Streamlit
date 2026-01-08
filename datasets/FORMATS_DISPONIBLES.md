# 📁 Formats de Fichiers Disponibles

## ✅ Formats Supportés par ChatAutoML-Bot

L'application supporte **4 formats de fichiers** selon le cahier des charges :

1. **CSV** (.csv) ✅
2. **Excel** (.xlsx, .xls) ⚠️ (nécessite openpyxl)
3. **Parquet** (.parquet) ✅
4. **JSON tabulaire** (.json) ✅

---

## 📊 Datasets Disponibles dans Tous les Formats

### Datasets Générés

Chaque dataset est disponible dans **4 formats** :

#### 1. Classification Binaire Équilibrée
- `binary_classification.csv`
- `binary_classification.xlsx` (si openpyxl installé)
- `binary_classification.parquet`
- `binary_classification.json`

#### 2. Classification Multiclasse
- `multiclass_classification.csv`
- `multiclass_classification.xlsx`
- `multiclass_classification.parquet`
- `multiclass_classification.json`

#### 3. Régression
- `regression.csv`
- `regression.xlsx`
- `regression.parquet`
- `regression.json`

#### 4. Avec Valeurs Manquantes
- `with_missing_values.csv`
- `with_missing_values.xlsx`
- `with_missing_values.parquet`
- `with_missing_values.json`

#### 5. Features Mixtes (Numériques + Catégorielles)
- `mixed_features.csv`
- `mixed_features.xlsx`
- `mixed_features.parquet`
- `mixed_features.json`

#### 6. Classification Déséquilibrée
- `imbalanced_binary.csv`
- `imbalanced_binary.xlsx`
- `imbalanced_binary.parquet`
- `imbalanced_binary.json`

#### 7. Dataset Petit (Tests Rapides)
- `small.csv`
- `small.xlsx`
- `small.parquet`
- `small.json`

### Datasets Réels

#### Iris
- `iris.csv`
- `iris.xlsx`
- `iris.parquet`
- `iris.json`

#### Titanic
- `titanic.csv`
- `titanic.xlsx`
- `titanic.parquet`
- `titanic.json`

---

## 🔧 Installation des Dépendances

Pour créer les fichiers Excel, installez `openpyxl` :

```bash
pip install openpyxl
```

Ou installez toutes les dépendances :

```bash
pip install -r requirements.txt
```

---

## 🚀 Comment Créer Tous les Formats

Exécutez le script de création :

```bash
python datasets/create_all_formats.py
```

Ce script va :
- ✅ Créer des datasets dans tous les formats
- ✅ Convertir les datasets existants
- ✅ Générer CSV, Excel, Parquet et JSON

---

## 📝 Format JSON Tabulaire

Le format JSON est une **liste de dictionnaires** comme spécifié dans le cahier des charges :

```json
[
  {
    "feature_1": 1.23,
    "feature_2": 4.56,
    "target": "class_A"
  },
  {
    "feature_1": 2.34,
    "feature_2": 5.67,
    "target": "class_B"
  }
]
```

---

## ✅ Vérification

Pour vérifier quels formats sont disponibles :

```bash
# Lister tous les fichiers par format
ls datasets/*.csv
ls datasets/*.xlsx
ls datasets/*.parquet
ls datasets/*.json
```

---

## 💡 Utilisation dans l'Application

1. **Lancez ChatAutoML-Bot** :
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Dans le menu latéral** :
   - Cliquez sur "📁 Charger un Dataset"
   - Sélectionnez **n'importe quel format** (CSV, Excel, Parquet, JSON)
   - L'application détectera automatiquement le format et chargera les données

3. **L'application supporte automatiquement** :
   - Détection du format par extension
   - Lecture avec pandas (read_csv, read_excel, read_parquet, read_json)
   - Gestion de tous les types de données

---

## 📊 Résumé

| Format | Extension | Statut | Notes |
|--------|-----------|--------|-------|
| CSV | .csv | ✅ Disponible | Format standard |
| Excel | .xlsx | ⚠️ Nécessite openpyxl | Format moderne Excel |
| Excel | .xls | ⚠️ Format obsolète | Utilisez .xlsx |
| Parquet | .parquet | ✅ Disponible | Format efficace |
| JSON | .json | ✅ Disponible | Liste de dictionnaires |

---

**Tous les formats sont prêts à être utilisés ! 🚀**


