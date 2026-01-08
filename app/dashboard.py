"""
Module pour créer un dashboard intuitif avec menu
"""
import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd


def create_final_summary(analysis: Dict[str, Any], 
                        preprocessing_info: Dict[str, Any],
                        resampling_info: Optional[Dict[str, Any]],
                        search_results: list,
                        selection_result: Dict[str, Any],
                        evaluation: Dict[str, Any],
                        task_type: str) -> str:
    """
    Crée un résumé complet du pipeline du début à la fin
    """
    summary = f"""# 📊 Résumé Complet du Pipeline AutoML

## 🎯 1. Objectif et Configuration Initiale

**Type de tâche** : {task_type.upper()}
**Dataset** : {analysis['shape']['rows']:,} lignes × {analysis['shape']['columns']} colonnes
**Colonne cible** : {analysis.get('target_info', {}).get('column', 'N/A')}

**📋 Caractéristiques du dataset** :
- **Colonnes numériques** : {len([c for c, info in analysis.get('columns', {}).items() if info.get('type') == 'numeric'])}
- **Colonnes catégorielles** : {len([c for c, info in analysis.get('columns', {}).items() if info.get('type') == 'categorical'])}
- **Valeurs manquantes** : {sum(info.get('count', 0) for info in analysis.get('missing_values', {}).values())} valeurs au total

---

## 🔧 2. Preprocessing (Prétraitement)

### 2.1 Séparation Features/Cible
- **Features (X)** : {preprocessing_info.get('n_features_before', 'N/A')} colonnes
- **Cible (y)** : 1 colonne ({analysis.get('target_info', {}).get('column', 'N/A')})

### 2.2 Gestion des Valeurs Manquantes
"""
    
    # Détails sur l'imputation
    missing_cols = [col for col, info in analysis.get('missing_values', {}).items() if info.get('count', 0) > 0]
    if missing_cols:
        summary += f"- **Colonnes avec valeurs manquantes** : {len(missing_cols)}\n"
        summary += "- **Stratégies appliquées** :\n"
        summary += "  - Numériques → Imputation par la **médiane**\n"
        summary += "  - Catégorielles → Imputation par **'Unknown'**\n"
        summary += "  - Booléennes → Imputation par le **mode**\n"
    else:
        summary += "- ✅ **Aucune valeur manquante** détectée\n"
    
    summary += f"""
### 2.3 Encodage des Variables Catégorielles
- **Méthode** : OneHotEncoding
- **Gestion des nouvelles catégories** : 'ignore' (ignorées lors de la prédiction)

### 2.4 Normalisation des Variables Numériques
- **Méthode** : StandardScaler (moyenne=0, écart-type=1)
- **Raison** : Nécessaire pour les modèles sensibles à l'échelle (SVM, KNN, régression linéaire)

### 2.5 Split Train/Test
- **Train** : {preprocessing_info.get('n_train', 'N/A')} échantillons ({preprocessing_info.get('train_pct', 'N/A')}%)
- **Test** : {preprocessing_info.get('n_test', 'N/A')} échantillons ({preprocessing_info.get('test_pct', 'N/A')}%)
- **Stratification** : {'✅ Appliquée' if task_type == 'classification' else 'N/A (régression)'}

### 2.6 Résultat du Preprocessing
- **Avant** : {preprocessing_info.get('n_features_before', 'N/A')} colonnes
- **Après** : {preprocessing_info.get('n_features_after', 'N/A')} features
- **Transformation** : Prêt pour l'entraînement ✅

---

## ⚖️ 3. Gestion du Déséquilibre (Classification uniquement)
"""
    
    if task_type == 'classification' and resampling_info:
        if resampling_info.get('applied'):
            summary += f"""**⚠️ Déséquilibre détecté et corrigé**

- **Distribution initiale** : {resampling_info.get('initial_distribution', 'N/A')}
- **Stratégie appliquée** : {resampling_info.get('strategy', 'N/A').upper()}
- **Échantillons avant** : {resampling_info.get('n_before', 'N/A')}
- **Échantillons après** : {resampling_info.get('n_after', 'N/A')}
- **Impact** : Les classes sont maintenant équilibrées pour un apprentissage équitable ✅
"""
        else:
            summary += """**✅ Classes équilibrées**

- Aucun rééchantillonnage nécessaire
- Les classes sont naturellement équilibrées
"""
    else:
        summary += "- N/A (Régression ou classes équilibrées)\n"
    
    summary += f"""
---

## 🔍 4. Recherche de Modèles et Optimisation

### 4.1 Modèles Testés
**{len(search_results)} modèles** ont été testés avec optimisation d'hyperparamètres :

"""
    
    for i, result in enumerate(search_results, 1):
        summary += f"{i}. **{result['model_name']}**\n"
        summary += f"   - Score : {result['best_score']:.4f}\n"
        summary += f"   - Temps : {result['elapsed_time']:.2f}s\n"
        if result.get('best_params'):
            summary += f"   - Meilleurs hyperparamètres : {result['best_params']}\n"
        summary += "\n"
    
    summary += f"""
### 4.2 Méthode d'Optimisation
- **Validation croisée** : {preprocessing_info.get('cv_folds', 5)}-fold
- **Méthode de recherche** : {preprocessing_info.get('search_method', 'grid').upper()}
- **Métrique principale** : {preprocessing_info.get('metric', 'auto')}

---

## 🏆 5. Sélection du Meilleur Modèle

**✅ Modèle retenu** : **{selection_result.get('best_model_name', 'N/A')}**

### 5.1 Performance
- **Score principal** : **{selection_result.get('best_score', 0):.4f}**
- **Basé sur** : Validation croisée {preprocessing_info.get('cv_folds', 5)}-fold

### 5.2 Hyperparamètres Optimaux
"""
    
    for param, value in selection_result.get('best_params', {}).items():
        summary += f"- **{param}** : {value}\n"
    
    summary += f"""
### 5.3 Pourquoi ce modèle ?
Ce modèle a été sélectionné car il présente la **meilleure performance** sur la métrique '{preprocessing_info.get('metric', 'auto')}' après optimisation exhaustive des hyperparamètres. Les hyperparamètres ont été choisis pour équilibrer la complexité du modèle et sa capacité de généralisation.

---

## 📈 6. Évaluation Complète

### 6.1 Performances sur Train vs Test
"""
    
    train_metrics = evaluation.get('train_metrics', {})
    test_metrics = evaluation.get('test_metrics', {})
    
    if task_type == 'classification':
        summary += f"""
| Métrique | Train | Test | Écart |
|----------|-------|------|-------|
| **Accuracy** | {train_metrics.get('accuracy', 0):.4f} | {test_metrics.get('accuracy', 0):.4f} | {abs(train_metrics.get('accuracy', 0) - test_metrics.get('accuracy', 0)):.4f} |
| **F1 Macro** | {train_metrics.get('f1_macro', 0):.4f} | {test_metrics.get('f1_macro', 0):.4f} | {abs(train_metrics.get('f1_macro', 0) - test_metrics.get('f1_macro', 0)):.4f} |
"""
        if 'roc_auc' in train_metrics:
            summary += f"| **ROC AUC** | {train_metrics.get('roc_auc', 0):.4f} | {test_metrics.get('roc_auc', 0):.4f} | {abs(train_metrics.get('roc_auc', 0) - test_metrics.get('roc_auc', 0)):.4f} |\n"
    else:
        summary += f"""
| Métrique | Train | Test | Écart |
|----------|-------|------|-------|
| **R²** | {train_metrics.get('r2', 0):.4f} | {test_metrics.get('r2', 0):.4f} | {abs(train_metrics.get('r2', 0) - test_metrics.get('r2', 0)):.4f} |
| **RMSE** | {train_metrics.get('rmse', 0):.4f} | {test_metrics.get('rmse', 0):.4f} | {abs(train_metrics.get('rmse', 0) - test_metrics.get('rmse', 0)):.4f} |
| **MAE** | {train_metrics.get('mae', 0):.4f} | {test_metrics.get('mae', 0):.4f} | {abs(train_metrics.get('mae', 0) - test_metrics.get('mae', 0)):.4f} |
"""
    
    summary += "\n### 6.2 Diagnostic de Généralisation\n"
    
    if evaluation.get('overfitting', {}).get('detected'):
        overfitting = evaluation['overfitting']
        summary += f"""⚠️ **OVERFITTING DÉTECTÉ**

- **Écart train/test** : {overfitting.get('gap', 0):.4f}
- **Score train** : {overfitting.get('train_score', 0):.4f}
- **Score test** : {overfitting.get('test_score', 0):.4f}
- **Interprétation** : Le modèle mémorise les données d'entraînement et ne généralise pas bien
- **Recommandation** : Augmenter la régularisation, réduire la complexité, ou collecter plus de données

"""
    elif evaluation.get('underfitting', {}).get('detected'):
        underfitting = evaluation['underfitting']
        summary += f"""⚠️ **UNDERFITTING DÉTECTÉ**

- **Score train** : {underfitting.get('train_score', 0):.4f}
- **Score test** : {underfitting.get('test_score', 0):.4f}
- **Interprétation** : Le modèle est trop simple pour capturer les patterns dans les données
- **Recommandation** : Utiliser un modèle plus complexe ou faire du feature engineering

"""
    else:
        summary += """✅ **GÉNÉRALISATION CORRECTE**

- Le modèle performe bien sur les données d'entraînement ET de test
- L'écart entre train et test est acceptable
- Le modèle généralise correctement aux nouvelles données

"""
    
    summary += f"""
---

## 💡 7. Insights et Recommandations

### 7.1 Points Forts
- ✅ Preprocessing complet et adapté au type de données
"""
    
    if task_type == 'classification' and resampling_info and resampling_info.get('applied'):
        summary += "- ✅ Déséquilibre des classes géré efficacement\n"
    
    summary += f"- ✅ {len(search_results)} modèles testés avec optimisation rigoureuse\n"
    summary += f"- ✅ Meilleur modèle sélectionné : {selection_result.get('best_model_name', 'N/A')}\n"
    
    summary += "\n### 7.2 Points d'Attention\n"
    
    # Vérifier les valeurs manquantes
    missing_cols = [col for col, info in analysis.get('missing_values', {}).items() 
                   if info.get('percentage', 0) > 10]
    if missing_cols:
        summary += f"- ⚠️ {len(missing_cols)} colonnes ont plus de 10% de valeurs manquantes\n"
    
    # Overfitting/underfitting
    if evaluation.get('overfitting', {}).get('detected'):
        summary += "- ⚠️ Overfitting détecté - nécessite attention\n"
    elif evaluation.get('underfitting', {}).get('detected'):
        summary += "- ⚠️ Underfitting détecté - modèle peut être amélioré\n"
    
    summary += "\n### 7.3 Pistes d'Amélioration\n"
    summary += "1. **Feature Engineering** : Créer de nouvelles features à partir des existantes\n"
    summary += "2. **Collecte de données** : Plus de données d'entraînement peuvent améliorer les performances\n"
    summary += "3. **Hyperparamètres** : Affiner davantage les hyperparamètres du meilleur modèle\n"
    summary += "4. **Modèles avancés** : Tester des modèles plus sophistiqués (XGBoost, LightGBM si pas déjà fait)\n"
    
    if evaluation.get('overfitting', {}).get('detected'):
        summary += "5. **Régularisation** : Augmenter la régularisation pour réduire l'overfitting\n"
    
    summary += "\n---\n\n"
    summary += "## ✅ Conclusion\n\n"
    summary += f"Le pipeline AutoML a été exécuté avec succès ! Le modèle **{selection_result.get('best_model_name', 'N/A')}** "
    summary += f"a été sélectionné avec un score de **{selection_result.get('best_score', 0):.4f}**. "
    summary += "Le modèle est prêt à être utilisé pour faire des prédictions sur de nouvelles données.\n\n"
    summary += "**🎯 Prochaines étapes** : Utiliser ce modèle pour faire des prédictions ou continuer à l'optimiser selon vos besoins."
    
    return summary

