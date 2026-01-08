"""
Module d'interface conversationnelle pour le chatbot
"""
import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import re


def show_welcome_message():
    """Affiche le message de bienvenue avec le menu - conforme à l'image"""
    # Récupérer le nom du chatbot depuis la config
    try:
        import config.config as cfg
        chatbot_name = cfg.CHATBOT_NAME
        chatbot_description = getattr(cfg, 'CHATBOT_DESCRIPTION', 'Votre assistant intelligent pour l\'AutoML sur données tabulaires')
        chatbot_icon = getattr(cfg, 'CHATBOT_ICON', '🤖')
    except (ImportError, AttributeError):
        chatbot_name = "ChatAutoML-Bot"
        chatbot_description = "Votre assistant intelligent pour l'AutoML sur données tabulaires"
        chatbot_icon = "🤖"
    
    welcome_text = f"""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <h1 style="color: #1f77b4; margin-bottom: 0.5rem; font-size: 2rem;">{chatbot_icon} {chatbot_name}</h1>
    <p style="font-size: 1.1rem; color: #666; margin: 0;">
        {chatbot_description}
    </p>
</div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 12px; color: white; margin: 2rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h2 style="color: white; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
        <span>🚀</span> Pipeline AutoML Complet
    </h2>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.2rem;">
        <div style="background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 8px; backdrop-filter: blur(10px);">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📊</div>
            <strong style="font-size: 1.1rem; display: block; margin-bottom: 0.3rem;">Analyse des données</strong>
            <small style="opacity: 0.9; font-size: 0.9rem;">Exploration et statistiques</small>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 8px; backdrop-filter: blur(10px);">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🔧</div>
            <strong style="font-size: 1.1rem; display: block; margin-bottom: 0.3rem;">Preprocessing</strong>
            <small style="opacity: 0.9; font-size: 0.9rem;">Nettoyage et préparation</small>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 8px; backdrop-filter: blur(10px);">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">⚖️</div>
            <strong style="font-size: 1.1rem; display: block; margin-bottom: 0.3rem;">Gestion du déséquilibre</strong>
            <small style="opacity: 0.9; font-size: 0.9rem;">SMOTE, oversampling</small>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 8px; backdrop-filter: blur(10px);">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🔍</div>
            <strong style="font-size: 1.1rem; display: block; margin-bottom: 0.3rem;">Recherche de modèles</strong>
            <small style="opacity: 0.9; font-size: 0.9rem;">Optimisation d'hyperparamètres</small>
        </div>
    </div>
</div>
"""
    return welcome_text


def detect_user_intent(message: str) -> Dict[str, Any]:
    """
    Détecte l'intention de l'utilisateur à partir de son message
    
    Returns:
        Dict avec 'intent', 'confidence', 'entities'
    """
    message_lower = message.lower().strip()

    # Normaliser un peu
    message_lower = " ".join(message_lower.split())
    
    # Salutations
    if any(word in message_lower for word in ['bonjour', 'salut', 'hello', 'hi', 'coucou']):
        return {'intent': 'greeting', 'confidence': 0.9, 'entities': {}}

    # Demande de statut / "vous avez fini ?"
    if any(phrase in message_lower for phrase in [
        "vous avez fini", "t'as fini", "tu as fini", "as-tu fini", "avez-vous fini",
        "vous avez terminé", "tu as terminé", "c'est fini", "c'est terminé", "c est fini", "c est termine",
        "c'est bon", "c est bon",
    ]):
        return {'intent': 'status_check', 'confidence': 0.9, 'entities': {}}

    # Revenir sur une étape / expliquer mieux
    if any(w in message_lower for w in [
        'explique', 'expliquer', 'réexplique', 'reexplique', 'détaille', 'detaille', 'plus de détail', 'plus de detail',
        'explain', 'clarify', 'details', 'detail', 'elaborate',
        'revenir', 'revient', 'reviens', 'revenez', 'retour', 'retourne', 'retournez', 'reprendre', 'reprends', 'reprenez',
    ]):
        entities: Dict[str, Any] = {}
        m = re.search(r"(étape|etape|step)\s*(\d)", message_lower)
        if m:
            try:
                entities['step'] = int(m.group(2))
            except Exception:
                pass

        # Mots-clés de sections
        if any(w in message_lower for w in ['analyse', 'dataset', 'données', 'donnees']):
            entities['section'] = 'analysis'
        if any(w in message_lower for w in ['prétraitement', 'pretraitement', 'preprocessing']):
            entities['section'] = 'preprocessing'
        if any(w in message_lower for w in ['déséquilibre', 'desequilibre', 'imbalance', 'smote', 'oversampling', 'undersampling']):
            entities['section'] = 'imbalance'
        if any(w in message_lower for w in ['comparaison', 'modèles', 'modeles', 'tableau', 'cv', 'grid', 'random search']):
            entities['section'] = 'model_comparison'
        if any(w in message_lower for w in ['évaluation', 'evaluation', 'train', 'test', 'overfitting', 'underfitting']):
            entities['section'] = 'evaluation'
        if any(w in message_lower for w in ['importance', 'variables', 'features', 'feature importance']):
            entities['section'] = 'feature_importance'
        if any(w in message_lower for w in ['résumé', 'resume', 'rapport', 'conclusion']):
            entities['section'] = 'final_summary'

        return {'intent': 'explain_step', 'confidence': 0.85, 'entities': entities}

    # Refaire / relancer une étape
    if any(w in message_lower for w in [
        'refais', 'refait', 'refaite', 'refaire', 'refaites', 'recommence', 'recommencer', 'recommencez',
        'relance', 'relancer', 'relancez', 'recalcule', 'recalculer', 'recalculez',
        're-exécute', 'reexecute', 'ré-exécute', 'réexecute'
    ]):
        entities: Dict[str, Any] = {}
        m = re.search(r"(étape|etape|step)\s*(\d)", message_lower)
        if m:
            try:
                entities['step'] = int(m.group(2))
            except Exception:
                pass

        if any(w in message_lower for w in ['analyse', 'dataset', 'données', 'donnees']):
            entities['section'] = 'analysis'
        if any(w in message_lower for w in ['prétraitement', 'pretraitement', 'preprocessing']):
            entities['section'] = 'preprocessing'
        if any(w in message_lower for w in ['déséquilibre', 'desequilibre', 'imbalance', 'smote', 'oversampling', 'undersampling']):
            entities['section'] = 'imbalance'
        if any(w in message_lower for w in ['comparaison', 'modèles', 'modeles', 'tableau', 'cv', 'grid', 'random search']):
            entities['section'] = 'model_comparison'
        if any(w in message_lower for w in ['évaluation', 'evaluation', 'train', 'test', 'overfitting', 'underfitting']):
            entities['section'] = 'evaluation'
        if any(w in message_lower for w in ['importance', 'variables', 'features', 'feature importance']):
            entities['section'] = 'feature_importance'
        if any(w in message_lower for w in ['résumé', 'resume', 'rapport', 'conclusion']):
            entities['section'] = 'final_summary'

        return {'intent': 'redo_step', 'confidence': 0.85, 'entities': entities}

    # Feedback / insatisfaction
    if any(phrase in message_lower for phrase in [
        "j'aime pas", "j aime pas", "je n'aime pas", "j'aime pas ça", "c'est nul", "c'est pas bien",
        "je n'aime pas", "je suis pas content", "je suis mécontent", "ça marche pas", "ça ne marche pas",
        "bug", "problème", "casse", "gâché",
    ]):
        return {'intent': 'feedback_negative', 'confidence': 0.85, 'entities': {}}

    # Reset / refaire / recommencer
    if any(phrase in message_lower for phrase in [
        "refais tout", "refaire tout", "recommencer tout", "reset", "réinitialiser", "reinitialiser",
        "nouvelle session", "tout supprimer", "remettre à zéro", "remettre a zero",
    ]):
        # Par défaut on comprend "reset session" (tout), mais on peut affiner via entities
        entities = {}
        if any(w in message_lower for w in ["pipeline", "automl", "modèle", "modele", "résultats", "resultats"]):
            entities['scope'] = 'pipeline'
            return {'intent': 'reset_pipeline', 'confidence': 0.9, 'entities': entities}
        entities['scope'] = 'session'
        return {'intent': 'reset_session', 'confidence': 0.9, 'entities': entities}

    # Changer le thème
    if any(word in message_lower for word in ["thème", "theme", "sombre", "clair", "dark", "light"]):
        entities = {}
        if any(w in message_lower for w in ["sombre", "dark"]):
            entities['theme'] = 'sombre'
        elif any(w in message_lower for w in ["clair", "light"]):
            entities['theme'] = 'clair'
        return {'intent': 'change_theme', 'confidence': 0.75, 'entities': entities}
    
    # Charger un dataset (éviter de capturer toutes les phrases qui mentionnent juste "dataset")
    if any(word in message_lower for word in ['charger', 'upload', 'importer', 'téléverser', 'televerser', 'fichier']) or (
        'dataset' in message_lower and any(v in message_lower for v in ['charger', 'upload', 'importer'])
    ):
        return {'intent': 'load_data', 'confidence': 0.9, 'entities': {}}
    
    # Preprocessing
    if any(word in message_lower for word in ['preprocessing', 'pretraitement', 'nettoyer', 'préparer', 'traiter']):
        return {'intent': 'preprocessing', 'confidence': 0.9, 'entities': {}}
    
    # Lancer AutoML (plus strict pour éviter de capturer "train/test" ou "modèle" dans une question)
    if (
        'automl' in message_lower
        or any(word in message_lower for word in ['lancer automl', 'lance automl', 'commencer automl', 'demarrer automl', 'démarrer automl'])
        or any(word in message_lower for word in ['entraîner', 'entrainer']) and any(word in message_lower for word in ['modèle', 'modele', 'pipeline'])
    ):
        return {'intent': 'run_automl', 'confidence': 0.85, 'entities': {}}
    
    # Questions sur les données
    if any(word in message_lower for word in ['données', 'dataset', 'colonnes', 'lignes', 'statistiques']):
        return {'intent': 'ask_about_data', 'confidence': 0.8, 'entities': {}}
    
    # Questions sur les modèles
    if any(word in message_lower for word in ['modèle', 'model', 'performance', 'score', 'résultat']):
        return {'intent': 'ask_about_models', 'confidence': 0.8, 'entities': {}}
    
    # Aide
    if any(word in message_lower for word in ['aide', 'help', 'commande', 'menu', 'que puis']):
        return {'intent': 'help', 'confidence': 0.9, 'entities': {}}
    
    # Question générale
    if '?' in message or any(word in message_lower for word in [
        'quoi', 'comment', 'pourquoi', 'explique',
        # Sous-questions pipeline (souvent sans "?")
        'normalisation', 'normalisation des', 'standardisation', 'scaling', 'standardscaler',
        'encodage', 'onehot', 'one-hot', 'imputation',
        'split', 'train/test', 'train test', 'stratification',
        'smote', 'oversampling', 'undersampling', 'déséquilibre', 'desequilibre',
        'cv', 'validation croisée', 'validation croisee',
        'hyperparam', 'hyperparamètre', 'hyperparametre', 'grid', 'random search'
    ]):
        return {'intent': 'question', 'confidence': 0.7, 'entities': {}}
    
    # Par défaut
    return {'intent': 'general', 'confidence': 0.5, 'entities': {}}


def generate_response(intent: Dict[str, Any], context: Dict[str, Any], 
                     llm_explainer, message: str = "") -> str:
    """Génère une réponse basée sur l'intention détectée"""
    
    intent_type = intent.get('intent', 'general')
    
    # Récupérer le nom du chatbot depuis la config
    try:
        import config.config as cfg
        chatbot_name = cfg.CHATBOT_NAME
        chatbot_icon = getattr(cfg, 'CHATBOT_ICON', '🤖')
    except (ImportError, AttributeError):
        chatbot_name = "ChatAutoML-Bot"
        chatbot_icon = "🤖"
    
    if intent_type == 'greeting':
        dataset = context.get('dataset')
        target_column = context.get('target_column')
        automl_done = bool(context.get('automl_done', False))
        selection_result = context.get('selection_result')
        
        if dataset is not None and target_column is not None and automl_done:
            best_name = None
            try:
                best_name = (selection_result or {}).get('best_model_name')
            except Exception:
                best_name = None
            return f"""Bonjour ! 👋

**État actuel** : AutoML terminé ✅
- Dataset : ✅ Chargé
- Colonne cible : {target_column}
- Meilleur modèle : {best_name or 'N/A'}

Tu peux me demander :
- "explique la comparaison des modèles"
- "explique l'évaluation"
- "reviens sur le prétraitement""" 

        if dataset is not None and target_column is not None:
            task_type = context.get('task_type', 'inconnu')
            return f"""Bonjour ! 👋

**État actuel** :
- Dataset : ✅ Chargé
- Colonne cible : {target_column}
- Type : {task_type}

**Prochaine étape** : Cliquez sur "🚀 Lancer AutoML" dans le menu latéral."""
        
        elif dataset is not None:
            return f"""Bonjour ! 👋

**Dataset chargé** ✅

**Prochaine étape** : Sélectionnez la colonne cible dans le menu latéral."""
        
        else:
            return f"""Bonjour ! 👋

Je suis {chatbot_name}.

**Pour commencer** : Chargez un fichier via le menu latéral (CSV, Excel, Parquet, JSON)."""
    
    elif intent_type == 'status_check':
        dataset = context.get('dataset')
        target_column = context.get('target_column')
        analysis_done = bool(context.get('analysis_done', False))
        preprocessing_done = bool(context.get('preprocessing_done', False))
        automl_done = bool(context.get('automl_done', False))

        lines = []
        lines.append(f"- Dataset : {'✅' if dataset is not None else '❌'}")
        lines.append(f"- Cible : {'✅ ' + str(target_column) if target_column else '❌'}")
        lines.append(f"- Analyse : {'✅' if analysis_done else '❌'}")
        lines.append(f"- Prétraitement : {'✅' if preprocessing_done else '❌'}")
        lines.append(f"- AutoML : {'✅' if automl_done else '❌'}")

        if automl_done:
            next_step = "Oui, c'est terminé ✅. Vous pouvez lire le **Résumé global AutoML** (en bas) ou me poser une question sur les résultats."
        elif preprocessing_done:
            next_step = "Pas encore. Prochaine étape : cliquez sur **🚀 Lancer l'AutoML** dans le menu latéral."
        elif analysis_done:
            next_step = "Pas encore. Prochaine étape : cliquez sur **🔧 Lancer le Prétraitement**."
        elif dataset is not None and target_column:
            next_step = "Pas encore. Prochaine étape : cliquez sur **🔍 Analyser le Dataset**."
        elif dataset is not None:
            next_step = "Pas encore. Prochaine étape : choisissez la **colonne cible** dans le menu latéral."
        else:
            next_step = "Pas encore. Prochaine étape : chargez un dataset via le menu latéral."

        return """📌 **État du pipeline**

""" + "\n".join(lines) + "\n\n" + next_step

    elif intent_type == 'explain_step':
        entities = intent.get('entities') or {}
        step = entities.get('step')
        section = entities.get('section')
        if step:
            return f"✅ D'accord. Je reviens sur <b>l'étape {step}/6</b> et je la ré-affiche avec les tableaux/graphes."
        if section:
            labels = {
                'analysis': 'Analyse du dataset',
                'preprocessing': 'Prétraitement',
                'imbalance': 'Déséquilibre',
                'model_comparison': 'Comparaison des modèles',
                'evaluation': 'Évaluation finale',
                'feature_importance': 'Importance des variables',
                'final_summary': 'Résumé global',
            }
            return f"✅ D'accord. Je reviens sur <b>{labels.get(section, section)}</b> et je la ré-affiche avec les tableaux/graphes."
        return "✅ D'accord. Dis-moi sur quelle partie tu veux revenir : analyse, prétraitement, déséquilibre, comparaison, évaluation, importance." 

    elif intent_type == 'redo_step':
        entities = intent.get('entities') or {}
        step = entities.get('step')
        section = entities.get('section')
        if step:
            return f"✅ D'accord. Je relance <b>l'étape {step}/6</b>."
        if section:
            labels = {
                'analysis': 'Analyse du dataset',
                'preprocessing': 'Prétraitement',
                'imbalance': 'Déséquilibre',
                'model_comparison': 'Comparaison des modèles',
                'evaluation': 'Évaluation finale',
                'feature_importance': 'Importance des variables',
                'final_summary': 'Résumé global',
            }
            return f"✅ D'accord. Je relance <b>{labels.get(section, section)}</b>."
        return "✅ D'accord. Dis-moi quelle partie je dois refaire : analyse, prétraitement, AutoML." 
    
    elif intent_type == 'load_data':
        return """📁 **Charger un dataset**

1. Utilisez le menu latéral
2. Section "📁 Charger un Dataset"
3. Sélectionnez votre fichier (CSV, Excel, Parquet, JSON)
4. Choisissez la colonne cible

**Formats** : CSV, Excel, Parquet, JSON"""
    
    elif intent_type == 'preprocessing':
        if context.get('dataset') is None:
            return """🔧 **Preprocessing**

Chargez d'abord un dataset, puis lancez AutoML. Le preprocessing sera automatique."""
        else:
            return """🔧 **Preprocessing**

Effectué automatiquement lors du lancement d'AutoML :
- Imputation des valeurs manquantes
- Encodage des catégorielles
- Normalisation des numériques
- Split train/test (80/20)"""
    
    elif intent_type == 'run_automl':
        if context.get('dataset') is None:
            return """🚀 **Lancer AutoML**

Chargez d'abord un dataset et sélectionnez la colonne cible."""
        elif context.get('target_column') is None:
            return """🚀 **Lancer AutoML**

Sélectionnez d'abord la colonne cible dans le menu latéral."""
        else:
            return """🚀 **Lancer AutoML**

Cliquez sur "🚀 Lancer AutoML" dans le menu latéral.

**Pipeline** :
1. Preprocessing
2. Gestion du déséquilibre
3. Recherche de modèles
4. Sélection du meilleur
5. Évaluation et explications"""
    
    elif intent_type == 'reset_pipeline':
        return (
            "✅ D'accord. Je peux **refaire le pipeline** (analyse → prétraitement → AutoML) sans supprimer le dataset.\n\n"
            "Confirme : tu veux\n"
            "- **A)** garder le dataset et juste refaire le pipeline\n"
            "- **B)** tout réinitialiser (dataset + chat + résultats)\n\n"
            "Réponds simplement `A` ou `B`."
        )

    elif intent_type == 'reset_session':
        return (
            "✅ D'accord. Tu veux que je **réinitialise toute la session** (dataset + cible + résultats + chat) ?\n\n"
            "Réponds : `oui` pour confirmer, ou `non` si tu veux seulement refaire le pipeline."
        )

    elif intent_type == 'change_theme':
        theme = (intent.get('entities') or {}).get('theme')
        if theme in ('sombre', 'clair'):
            return f"✅ D'accord. Je passe le thème en **{theme}**."
        return "✅ D'accord. Tu veux le thème **sombre** ou **clair** ?"

    elif intent_type == 'feedback_negative':
        return (
            "Je comprends. Dis-moi exactement ce que tu n'aimes pas et je corrige.\n\n"
            "Pour aller vite, réponds avec **1 chiffre** :\n"
            "1) C'est trop lent\n"
            "2) Upload / cible ne marche pas\n"
            "3) AutoML / résultats pas clairs\n"
            "4) UI / thème / design\n"
            "5) Chat ne comprend pas mes phrases\n\n"
            "Puis ajoute une phrase (optionnel)."
        )

    elif intent_type == 'help':
        return """📚 **Aide**

**Commandes** :
- "Bonjour" - Bienvenue
- "Charger un dataset" - Instructions
- "Lancer AutoML" - Démarrer le pipeline
- "Aide" - Cette aide

**Questions possibles** :
- Valeurs manquantes ?
- Déséquilibre ?
- Meilleur modèle ?
- Overfitting ?
- Features importantes ?

**Menu latéral** : Charger dataset, Sélectionner cible, Lancer AutoML"""
    
    elif intent_type == 'ask_about_data':
        if context.get('analysis'):
            return llm_explainer.answer_question(message, context)
        else:
            return """📊 **Questions sur les données**

Chargez d'abord un dataset via le menu latéral."""
    
    elif intent_type == 'ask_about_models':
        if context.get('selection_result'):
            return llm_explainer.answer_question(message, context)
        else:
            return """🤖 **Questions sur les modèles**

Lancez d'abord AutoML via le menu latéral."""
    
    elif intent_type == 'question':
        # Utiliser le LLM explainer pour répondre
        return llm_explainer.answer_question(message, context)
    
    else:
        # Fallback intelligent : tenter de répondre en langage naturel
        try:
            if llm_explainer is not None and (message or '').strip():
                natural = llm_explainer.answer_question(message, context)
                if isinstance(natural, str) and natural.strip():
                    return natural
        except Exception:
            pass

        dataset = context.get('dataset')
        if dataset is None:
            return "Je n'ai pas compris. Pour commencer, charge un dataset via le menu latéral, puis choisis la colonne cible."

        return "Je n'ai pas compris. Reformule en 1 phrase (ex: 'explique la normalisation', 'refais le prétraitement', 'reviens sur la comparaison')."

