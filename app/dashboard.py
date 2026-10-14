"""
Module pour créer un dashboard intuitif avec menu
"""
import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd
from app.visualizations import format_model_name_fr


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
    search_method = preprocessing_info.get('search_method', 'grid')
    search_method_label = {
        'grid': 'recherche par grille',
        'random': 'recherche aléatoire',
        'bayesian': 'recherche bayésienne',
    }.get(str(search_method).lower(), str(search_method))
    task_label = {'classification': 'classification', 'regression': 'régression'}.get(task_type, task_type)
    metric = preprocessing_info.get('metric', 'auto')
    metric_label = {
        'accuracy': 'exactitude',
        'f1': 'score F1',
        'f1_macro': 'score F1 macro',
        'f1_weighted': 'score F1 pondéré',
        'roc_auc': 'aire sous la courbe ROC',
        'r2': 'coefficient de détermination (R²)',
        'rmse': 'racine de l’erreur quadratique moyenne',
        'neg_root_mean_squared_error': 'racine de l’erreur quadratique moyenne',
        'mae': 'erreur absolue moyenne',
    }.get(str(metric).lower(), str(metric))
    resampling_strategy = (resampling_info or {}).get('strategy', 'N/A')
    resampling_label = {
        'smote': 'SMOTE',
        'random_over': 'suréchantillonnage aléatoire',
        'random_under': 'sous-échantillonnage aléatoire',
        'adasyn': 'ADASYN',
    }.get(str(resampling_strategy).lower(), str(resampling_strategy))
    summary = f"""# 📊 Synthèse globale du parcours AutoML

## 🎯 1. Objectif et Configuration Initiale

**Type de tâche** : {task_label}
**Jeu de données** : {analysis['shape']['rows']:,} lignes × {analysis['shape']['columns']} colonnes
**Colonne cible** : {analysis.get('target_info', {}).get('column', 'N/A')}

**📋 Caractéristiques du jeu de données** :
- **Colonnes numériques** : {len([c for c, info in analysis.get('columns', {}).items() if info.get('type') == 'numeric'])}
- **Colonnes catégorielles** : {len([c for c, info in analysis.get('columns', {}).items() if info.get('type') == 'categorical'])}
- **Valeurs manquantes** : {sum(info.get('count', 0) for info in analysis.get('missing_values', {}).values())} valeurs au total

---

## 🔧 2. Prétraitement

### 2.1 Séparation des variables et de la cible
- **Variables (X)** : {preprocessing_info.get('n_features_before', 'N/A')} colonnes
- **Cible (y)** : 1 colonne ({analysis.get('target_info', {}).get('column', 'N/A')})

### 2.2 Gestion des Valeurs Manquantes
"""
    
    # Détails sur l'imputation
    missing_cols = [col for col, info in analysis.get('missing_values', {}).items() if info.get('count', 0) > 0]
    if missing_cols:
        summary += f"- **Colonnes avec valeurs manquantes** : {len(missing_cols)}\n"
        summary += "- **Stratégies appliquées** :\n"
        summary += "  - Numériques → Imputation par la **médiane**\n"
        summary += "  - Catégorielles → Imputation par **« inconnue »**\n"
        summary += "  - Booléennes → Imputation par le **mode**\n"
    else:
        summary += "- ✅ **Aucune valeur manquante** détectée\n"
    
    summary += f"""
### 2.3 Encodage des variables catégorielles
- **Méthode** : encodage one-hot
- **Gestion des nouvelles catégories** : ignorées lors de la prédiction

### 2.4 Normalisation des variables numériques
- **Méthode** : standardisation (moyenne = 0, écart-type = 1)
- **Raison** : Nécessaire pour les modèles sensibles à l'échelle (SVM, KNN, régression linéaire)

### 2.5 Séparation entre entraînement et test
- **Entraînement** : {preprocessing_info.get('n_train', 'N/A')} échantillons ({preprocessing_info.get('train_pct', 'N/A')} %)
- **Test** : {preprocessing_info.get('n_test', 'N/A')} échantillons ({preprocessing_info.get('test_pct', 'N/A')}%)
- **Stratification** : {'✅ Appliquée' if task_type == 'classification' else 'N/A (régression)'}

### 2.6 Résultat du prétraitement
- **Avant** : {preprocessing_info.get('n_features_before', 'N/A')} colonnes
- **Après** : {preprocessing_info.get('n_features_after', 'N/A')} variables
- **Transformation** : Prêt pour l'entraînement ✅

---

## ⚖️ 3. Gestion du Déséquilibre (Classification uniquement)
"""
    
    if task_type == 'classification' and resampling_info:
        if resampling_info.get('applied'):
            summary += f"""**⚠️ Déséquilibre détecté et corrigé**

- **Distribution initiale** : {resampling_info.get('initial_distribution', 'N/A')}
- **Stratégie appliquée** : {resampling_label}
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
        summary += f"{i}. **{format_model_name_fr(result['model_name'])}**\n"
        summary += f"   - Score : {result['best_score']:.4f}\n"
        summary += f"   - Temps : {result['elapsed_time']:.2f}s\n"
        if result.get('best_params'):
            summary += f"   - Meilleurs hyperparamètres : {result['best_params']}\n"
        summary += "\n"
    
    summary += f"""
### 4.2 Méthode d'Optimisation
- **Validation croisée** : {preprocessing_info.get('cv_folds', 5)} plis
- **Méthode de recherche** : {search_method_label}
- **Indicateur principal** : {metric_label}

---

## 🏆 5. Sélection du Meilleur Modèle

**✅ Modèle retenu** : **{format_model_name_fr(selection_result.get('best_model_name', 'N/A'))}**

### 5.1 Performance
- **Score principal** : **{selection_result.get('best_score', 0):.4f}**
- **Basé sur** : validation croisée à {preprocessing_info.get('cv_folds', 5)} plis

### 5.2 Hyperparamètres Optimaux
"""
    
    for param, value in selection_result.get('best_params', {}).items():
        summary += f"- **{param}** : {value}\n"
    
    summary += f"""
### 5.3 Pourquoi ce modèle ?
Ce modèle a été sélectionné car il présente la **meilleure performance** sur l’indicateur « {metric_label} » après optimisation des hyperparamètres. Ceux-ci ont été choisis pour équilibrer la complexité du modèle et sa capacité de généralisation.

---

## 📈 6. Évaluation Complète

### 6.1 Performances sur les données d’entraînement et de test
"""
    
    train_metrics = evaluation.get('train_metrics', {})
    test_metrics = evaluation.get('test_metrics', {})
    
    if task_type == 'classification':
        summary += f"""
| Indicateur | Entraînement | Test | Écart |
|----------|-------|------|-------|
| **Exactitude** | {train_metrics.get('accuracy', 0):.4f} | {test_metrics.get('accuracy', 0):.4f} | {abs(train_metrics.get('accuracy', 0) - test_metrics.get('accuracy', 0)):.4f} |
| **Score F1 macro** | {train_metrics.get('f1_macro', 0):.4f} | {test_metrics.get('f1_macro', 0):.4f} | {abs(train_metrics.get('f1_macro', 0) - test_metrics.get('f1_macro', 0)):.4f} |
"""
        if 'roc_auc' in train_metrics:
            summary += f"| **Aire sous la courbe ROC** | {train_metrics.get('roc_auc', 0):.4f} | {test_metrics.get('roc_auc', 0):.4f} | {abs(train_metrics.get('roc_auc', 0) - test_metrics.get('roc_auc', 0)):.4f} |\n"
    else:
        summary += f"""
| Indicateur | Entraînement | Test | Écart |
|----------|-------|------|-------|
| **R²** | {train_metrics.get('r2', 0):.4f} | {test_metrics.get('r2', 0):.4f} | {abs(train_metrics.get('r2', 0) - test_metrics.get('r2', 0)):.4f} |
| **Racine de l’erreur quadratique moyenne** | {train_metrics.get('rmse', 0):.4f} | {test_metrics.get('rmse', 0):.4f} | {abs(train_metrics.get('rmse', 0) - test_metrics.get('rmse', 0)):.4f} |
| **Erreur absolue moyenne** | {train_metrics.get('mae', 0):.4f} | {test_metrics.get('mae', 0):.4f} | {abs(train_metrics.get('mae', 0) - test_metrics.get('mae', 0)):.4f} |
"""
    
    summary += "\n### 6.2 Diagnostic de Généralisation\n"
    
    if evaluation.get('overfitting', {}).get('detected'):
        overfitting = evaluation['overfitting']
        summary += f"""⚠️ **SURAPPRENTISSAGE DÉTECTÉ**

- **Écart entraînement/test** : {overfitting.get('gap', 0):.4f}
- **Score d’entraînement** : {overfitting.get('train_score', 0):.4f}
- **Score test** : {overfitting.get('test_score', 0):.4f}
- **Interprétation** : Le modèle mémorise les données d'entraînement et ne généralise pas bien
- **Recommandation** : Augmenter la régularisation, réduire la complexité, ou collecter plus de données

"""
    elif evaluation.get('underfitting', {}).get('detected'):
        underfitting = evaluation['underfitting']
        summary += f"""⚠️ **SOUS-APPRENTISSAGE DÉTECTÉ**

- **Score d’entraînement** : {underfitting.get('train_score', 0):.4f}
- **Score test** : {underfitting.get('test_score', 0):.4f}
- **Interprétation** : Le modèle est trop simple pour représenter les tendances des données
- **Recommandation** : Utiliser un modèle plus complexe ou créer de nouvelles variables

"""
    else:
        summary += """✅ **GÉNÉRALISATION CORRECTE**

- Le modèle fonctionne bien sur les données d'entraînement et de test
- L'écart entre l'entraînement et le test est acceptable
- Le modèle généralise correctement aux nouvelles données

"""
    
    summary += f"""
---

## 💡 7. Constats et recommandations

### 7.1 Points Forts
- ✅ Prétraitement complet et adapté au type de données
"""
    
    if task_type == 'classification' and resampling_info and resampling_info.get('applied'):
        summary += "- ✅ Déséquilibre des classes géré efficacement\n"
    
    summary += f"- ✅ {len(search_results)} modèles testés avec optimisation rigoureuse\n"
    summary += f"- ✅ Meilleur modèle sélectionné : {format_model_name_fr(selection_result.get('best_model_name', 'N/A'))}\n"
    
    summary += "\n### 7.2 Points d'Attention\n"
    
    # Vérifier les valeurs manquantes
    missing_cols = [col for col, info in analysis.get('missing_values', {}).items() 
                   if info.get('percentage', 0) > 10]
    if missing_cols:
        summary += f"- ⚠️ {len(missing_cols)} colonnes ont plus de 10% de valeurs manquantes\n"
    
    # Overfitting/underfitting
    if evaluation.get('overfitting', {}).get('detected'):
        summary += "- ⚠️ Surapprentissage détecté — une attention est nécessaire\n"
    elif evaluation.get('underfitting', {}).get('detected'):
        summary += "- ⚠️ Sous-apprentissage détecté — le modèle peut être amélioré\n"
    
    summary += "\n### 7.3 Pistes d'Amélioration\n"
    summary += "1. **Création de variables** : concevoir de nouvelles variables à partir des variables existantes\n"
    summary += "2. **Collecte de données** : Plus de données d'entraînement peuvent améliorer les performances\n"
    summary += "3. **Hyperparamètres** : Affiner davantage les hyperparamètres du meilleur modèle\n"
    summary += "4. **Modèles avancés** : Tester des modèles plus sophistiqués (XGBoost, LightGBM si pas déjà fait)\n"
    
    if evaluation.get('overfitting', {}).get('detected'):
        summary += "5. **Régularisation** : augmenter la régularisation pour réduire le surapprentissage\n"
    
    summary += "\n---\n\n"
    summary += "## ✅ Conclusion\n\n"
    summary += f"Le parcours AutoML a été exécuté avec succès ! Le modèle **{format_model_name_fr(selection_result.get('best_model_name', 'N/A'))}** "
    summary += f"a été sélectionné avec un score de **{selection_result.get('best_score', 0):.4f}**. "
    summary += "Le modèle est prêt à être utilisé pour faire des prédictions sur de nouvelles données.\n\n"
    summary += "**🎯 Prochaines étapes** : Utiliser ce modèle pour faire des prédictions ou continuer à l'optimiser selon vos besoins."
    
    return summary
