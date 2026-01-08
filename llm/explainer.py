"""
Module LLM pour générer des explications en langage naturel
"""
from typing import Dict, Any, Optional
import json


class LLMExplainer:
    """Génère des explications en langage naturel via LLM"""
    
    def __init__(self, llm_provider: str = 'ollama', model_name: Optional[str] = None, 
                 ollama_base_url: Optional[str] = None):
        """
        Args:
            llm_provider: 'ollama', 'openai', 'mistral', 'huggingface', ou 'template'
            model_name: Nom du modèle Ollama (ex: 'llama2', 'mistral', 'codellama')
            ollama_base_url: URL de base pour Ollama (défaut: 'http://localhost:11434')
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url or 'http://localhost:11434'
        self._setup_llm()
    
    def _setup_llm(self):
        """Configure le client LLM"""
        # Par défaut, on utilise une approche template-based
        self.use_api = False
        self.client = None
        
        if self.llm_provider == 'ollama':
            try:
                import ollama
                self.client = ollama
                self.use_api = True
                # Si model_name n'est pas spécifié, utiliser un modèle par défaut
                if not self.model_name:
                    self.model_name = 'llama2'  # Modèle par défaut
            except ImportError:
                # Essayer avec requests si ollama n'est pas installé
                try:
                    import requests
                    self.requests = requests
                    self.use_api = True
                    if not self.model_name:
                        self.model_name = 'llama2'
                except ImportError:
                    pass
        elif self.llm_provider == 'openai':
            try:
                import openai
                self.client = openai
                self.use_api = True
            except ImportError:
                pass
        elif self.llm_provider == 'mistral':
            try:
                from mistralai import Mistral
                # Note: api_key devrait être passé en paramètre si utilisé
                self.client = Mistral()
                self.use_api = True
            except ImportError:
                pass
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Appelle Ollama pour générer une réponse"""
        if not self.use_api:
            return ""
        
        try:
            # Essayer avec la bibliothèque ollama (si installée)
            if self.client and hasattr(self.client, 'chat'):
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                messages.append({'role': 'user', 'content': prompt})
                
                response = self.client.chat(model=self.model_name, messages=messages)
                if isinstance(response, dict):
                    return response.get('message', {}).get('content', '')
                elif hasattr(response, 'message'):
                    return response.message.content
                return str(response)
            elif hasattr(self, 'requests'):
                # Utiliser l'API REST d'Ollama (méthode alternative)
                url = f"{self.ollama_base_url}/api/chat"
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                messages.append({'role': 'user', 'content': prompt})
                
                payload = {
                    'model': self.model_name,
                    'messages': messages,
                    'stream': False
                }
                response = self.requests.post(url, json=payload, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    return result.get('message', {}).get('content', '')
                else:
                    # Essayer avec l'ancienne API /api/generate
                    url = f"{self.ollama_base_url}/api/generate"
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    payload = {
                        'model': self.model_name,
                        'prompt': full_prompt,
                        'stream': False
                    }
                    response = self.requests.post(url, json=payload, timeout=120)
                    if response.status_code == 200:
                        return response.json().get('response', '')
                    return ""
        except Exception as e:
            print(f"Erreur lors de l'appel à Ollama: {e}")
            import traceback
            traceback.print_exc()
            return ""
        
        return ""
    
    def explain_dataset_analysis(self, analysis: Dict[str, Any]) -> str:
        """Explique l'analyse du dataset"""
        # Préparer les données pour le prompt
        data_summary = {
            'rows': analysis['shape']['rows'],
            'columns': analysis['shape']['columns'],
            'column_info': {col: {
                'type': info['type'],
                'dtype': info['dtype'],
                'missing_pct': info.get('missing_percentage', info.get('percentage', 0))
            } for col, info in analysis['columns'].items()}
        }
        
        if analysis.get('target_info'):
            data_summary['target'] = analysis['target_info']
        
        # Si Ollama est disponible, l'utiliser pour générer une explication
        if self.use_api and self.llm_provider == 'ollama':
            prompt = f"""Analyse ce dataset et explique-le de manière claire et professionnelle en français.

Informations du dataset:
- Dimensions: {data_summary['rows']} lignes × {data_summary['columns']} colonnes
- Colonnes: {json.dumps(data_summary['column_info'], indent=2, ensure_ascii=False)}
"""
            if 'target' in data_summary:
                prompt += f"- Colonne cible: {data_summary['target']['column']} (type: {data_summary['target']['type']}, valeurs uniques: {data_summary['target']['unique_count']})\n"
                if not data_summary['target'].get('is_balanced'):
                    prompt += "- ⚠️ Le dataset est déséquilibré\n"
            
            prompt += "\nGénère une explication claire et structurée en markdown."
            
            system_prompt = "Tu es un expert en Machine Learning et Data Science. Tu expliques les analyses de données de manière claire et professionnelle en français."
            
            llm_response = self._call_ollama(prompt, system_prompt)
            if llm_response:
                return llm_response
        
        # Fallback sur template si LLM non disponible
        summary = f"""## Analyse du Dataset

**Dimensions** : {analysis['shape']['rows']} lignes × {analysis['shape']['columns']} colonnes

**Colonnes** :
"""
        for col, info in analysis['columns'].items():
            missing = info.get('missing_percentage', info.get('percentage', 0))
            summary += f"- **{col}** : {info['type']} ({info['dtype']})"
            if missing > 0:
                summary += f" - {missing:.1f}% de valeurs manquantes"
            summary += "\n"
        
        if analysis.get('target_info'):
            target = analysis['target_info']
            summary += f"\n**Colonne cible** : {target['column']}\n"
            summary += f"- Type détecté : {target['type']}\n"
            summary += f"- Nombre de classes/valeurs uniques : {target['unique_count']}\n"
            if not target['is_balanced']:
                summary += "⚠️ **Attention** : Le dataset est déséquilibré\n"
        
        return summary
    
    def explain_task_detection(self, task_type: str, confidence: str = "élevée") -> str:
        """Explique la détection du type de tâche"""
        if task_type == 'classification':
            return f"""## Type de tâche détecté : **Classification**

La colonne cible semble être une variable catégorielle ou discrète. 
Le système va entraîner des modèles de classification pour prédire la classe de chaque échantillon.

Confiance : {confidence}
"""
        else:
            return f"""## Type de tâche détecté : **Régression**

La colonne cible semble être une variable numérique continue.
Le système va entraîner des modèles de régression pour prédire une valeur numérique.

Confidence : {confidence}
"""
    
    def explain_model_selection(self, selection_result: Dict[str, Any], 
                               evaluation: Dict[str, Any],
                               task_type: str) -> str:
        """Explique la sélection du meilleur modèle"""
        model_name = selection_result['best_model_name']
        best_score = selection_result['best_score']
        
        # Si Ollama est disponible, l'utiliser
        if self.use_api and self.llm_provider == 'ollama':
            comparison_data = ""
            if 'comparison_summary' in selection_result:
                df = selection_result['comparison_summary']
                comparison_data = df.to_string(index=False)
            
            overfitting_info = ""
            if evaluation.get('overfitting', {}).get('detected'):
                overfitting_info = f"Overfitting détecté: train={evaluation['overfitting']['train_score']:.4f}, test={evaluation['overfitting']['test_score']:.4f}, écart={evaluation['overfitting']['gap']:.4f}"
            
            underfitting_info = ""
            if evaluation.get('underfitting', {}).get('detected'):
                underfitting_info = "Underfitting détecté"
            
            test_metrics = evaluation.get('test_metrics', {})
            
            prompt = f"""Explique les résultats de sélection du meilleur modèle de Machine Learning en français.

Meilleur modèle: {model_name}
Score principal: {best_score:.4f}
Type de tâche: {task_type}

Comparaison des modèles:
{comparison_data}

Métriques de test: {json.dumps(test_metrics, indent=2, ensure_ascii=False)}
{overfitting_info}
{underfitting_info}

Génère une explication claire, structurée en markdown, qui explique pourquoi ce modèle a été choisi, compare avec les autres, et commente les performances."""
            
            system_prompt = "Tu es un expert en Machine Learning. Tu expliques les résultats de modèles de manière claire et professionnelle en français."
            
            llm_response = self._call_ollama(prompt, system_prompt)
            if llm_response:
                return llm_response
        
        # Fallback sur template
        explanation = f"""## Meilleur modèle sélectionné : **{model_name}**

**Score principal** : {best_score:.4f}

### Comparaison avec les autres modèles

"""
        
        # Ajouter le tableau comparatif
        if 'comparison_summary' in selection_result:
            df = selection_result['comparison_summary']
            explanation += df.to_string(index=False) + "\n\n"
        
        # Informations sur overfitting/underfitting
        if evaluation.get('overfitting', {}).get('detected'):
            explanation += "⚠️ **Overfitting détecté** : Le modèle performe bien sur les données d'entraînement mais moins bien sur les données de test.\n"
            explanation += "   - Score train : {:.4f}\n".format(evaluation['overfitting']['train_score'])
            explanation += "   - Score test : {:.4f}\n".format(evaluation['overfitting']['test_score'])
            explanation += "   - Écart : {:.4f}\n\n".format(evaluation['overfitting']['gap'])
        
        if evaluation.get('underfitting', {}).get('detected'):
            explanation += "⚠️ **Underfitting détecté** : Le modèle ne performe pas bien sur les données d'entraînement et de test.\n"
            explanation += "   Cela suggère que le modèle est trop simple pour capturer les patterns dans les données.\n\n"
        
        # Métriques détaillées
        test_metrics = evaluation.get('test_metrics', {})
        explanation += "### Performances sur les données de test\n\n"
        
        if task_type == 'classification':
            explanation += f"- **Accuracy** : {test_metrics.get('accuracy', 0):.4f}\n"
            explanation += f"- **F1 Macro** : {test_metrics.get('f1_macro', 0):.4f}\n"
            explanation += f"- **F1 Weighted** : {test_metrics.get('f1_weighted', 0):.4f}\n"
            if 'roc_auc' in test_metrics:
                explanation += f"- **ROC AUC** : {test_metrics['roc_auc']:.4f}\n"
        else:
            explanation += f"- **R²** : {test_metrics.get('r2', 0):.4f}\n"
            explanation += f"- **RMSE** : {test_metrics.get('rmse', 0):.4f}\n"
            explanation += f"- **MAE** : {test_metrics.get('mae', 0):.4f}\n"
        
        return explanation
    
    def explain_feature_importance(self, feature_importance: Dict[str, float], 
                                   top_n: int = 10) -> str:
        """Explique l'importance des features"""
        if not feature_importance:
            return "L'importance des features n'est pas disponible pour ce modèle.\n"
        
        # Trier par importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        explanation = f"### Top {top_n} features les plus importantes\n\n"
        for i, (feature, importance) in enumerate(top_features, 1):
            explanation += f"{i}. **{feature}** : {importance:.4f}\n"
        
        return explanation
    
    def generate_improvement_suggestions(self, analysis: Dict[str, Any],
                                        evaluation: Dict[str, Any],
                                        task_type: str) -> str:
        """Génère des suggestions d'amélioration"""
        suggestions = "### Pistes d'amélioration\n\n"
        
        # Vérifier les valeurs manquantes
        missing_cols = [col for col, info in analysis.get('missing_values', {}).items() 
                       if info['percentage'] > 10]
        if missing_cols:
            suggestions += f"- **Valeurs manquantes** : {len(missing_cols)} colonnes ont plus de 10% de valeurs manquantes. "
            suggestions += "Considérez des stratégies d'imputation plus sophistiquées.\n"
        
        # Overfitting
        if evaluation.get('overfitting', {}).get('detected'):
            suggestions += "- **Overfitting** : Essayez d'augmenter la régularisation, réduire la complexité du modèle, "
            suggestions += "ou collecter plus de données.\n"
        
        # Underfitting
        if evaluation.get('underfitting', {}).get('detected'):
            suggestions += "- **Underfitting** : Le modèle est peut-être trop simple. Essayez des modèles plus complexes "
            suggestions += "ou feature engineering.\n"
        
        # Déséquilibre
        if analysis.get('target_info', {}).get('is_balanced') == False:
            suggestions += "- **Déséquilibre des classes** : Le rééchantillonnage a été appliqué, mais vous pourriez "
            suggestions += "essayer d'autres stratégies ou utiliser des métriques plus adaptées.\n"
        
        # Feature engineering
        suggestions += "- **Feature engineering** : Créez de nouvelles features à partir des existantes, "
        suggestions += "ou supprimez les features peu importantes.\n"
        
        suggestions += "- **Plus de données** : Collecter plus de données d'entraînement peut améliorer les performances.\n"
        
        return suggestions
    
    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """
        Répond à une question de l'utilisateur en utilisant le contexte
        
        Args:
            question: Question de l'utilisateur
            context: Contexte contenant les informations sur le dataset, modèles, etc.
        """
        # Si Ollama est disponible, l'utiliser pour répondre de manière plus naturelle
        if self.use_api and self.llm_provider == 'ollama':
            # Préparer le contexte pour le prompt
            context_str = "Contexte disponible:\n"
            analysis = context.get('analysis')
            if analysis is not None:
                try:
                    context_str += f"- Analyse du dataset: {json.dumps(analysis, indent=2, ensure_ascii=False, default=str)}\n"
                except (TypeError, ValueError):
                    pass
            
            task_type = context.get('task_type')
            if task_type is not None:
                context_str += f"- Type de tâche: {task_type}\n"
            selection_result = context.get('selection_result')
            if selection_result is not None:
                try:
                    context_str += f"- Meilleur modèle: {selection_result.get('best_model_name', 'N/A')}\n"
                except (AttributeError, TypeError):
                    pass
            
            evaluation = context.get('evaluation')
            if evaluation is not None:
                try:
                    context_str += f"- Évaluation: {json.dumps(evaluation, indent=2, ensure_ascii=False, default=str)}\n"
                except (TypeError, ValueError):
                    pass
            
            prompt = f"""{context_str}

Question de l'utilisateur: {question}

Réponds à la question de manière claire, précise et professionnelle en français. Utilise les informations du contexte si disponibles."""
            
            system_prompt = "Tu es un assistant expert en Machine Learning et Data Science. Tu réponds aux questions de manière claire, précise et professionnelle en français."
            
            llm_response = self._call_ollama(prompt, system_prompt)
            if llm_response:
                return llm_response
        
        # Fallback sur les réponses template
        question_lower = question.lower()
        
        # Questions sur les données
        if any(word in question_lower for word in ['données', 'dataset', 'colonnes', 'lignes']):
            if 'analysis' in context and context['analysis'] is not None:
                return self.explain_dataset_analysis(context['analysis'])
            return "Les informations sur les données ne sont pas encore disponibles."
        
        # Questions sur le déséquilibre
        if any(word in question_lower for word in ['déséquilibre', 'équilibré', 'balance']):
            if 'analysis' in context and context['analysis'] is not None and context['analysis'].get('target_info'):
                target_info = context['analysis']['target_info']
                if not target_info.get('is_balanced'):
                    return f"Oui, le dataset est déséquilibré. La classe majoritaire représente plus de 70% des données. Distribution : {target_info.get('distribution', {})}"
                return "Le dataset semble équilibré."
            return "Je n'ai pas encore analysé le déséquilibre des classes."
        
        # Questions sur les valeurs manquantes
        if any(word in question_lower for word in ['manquant', 'missing', 'na', 'null']):
            if 'analysis' in context and context['analysis'] is not None:
                missing = context['analysis'].get('missing_values', {})
                cols_with_missing = {k: v for k, v in missing.items() if v['count'] > 0}
                if cols_with_missing:
                    return f"Colonnes avec valeurs manquantes : {cols_with_missing}"
                return "Aucune valeur manquante détectée dans le dataset."
            return "Je n'ai pas encore analysé les valeurs manquantes."
        
        # Questions sur les modèles
        if any(word in question_lower for word in ['modèle', 'model', 'performance', 'score']):
            if 'selection_result' in context and context['selection_result'] is not None:
                return self.explain_model_selection(
                    context['selection_result'],
                    context.get('evaluation', {}),
                    context.get('task_type', 'classification')
                )
            return "Les modèles n'ont pas encore été entraînés."
        
        # Questions sur overfitting/underfitting
        if 'overfitting' in question_lower or 'surajustement' in question_lower:
            if 'evaluation' in context and context['evaluation'] is not None:
                overfitting = context['evaluation'].get('overfitting', {})
                if overfitting and overfitting.get('detected'):
                    return f"Oui, de l'overfitting a été détecté. Écart entre train et test : {overfitting.get('gap', 0):.4f}"
                return "Aucun overfitting significatif détecté."
            return "Je n'ai pas encore évalué l'overfitting."
        
        if 'underfitting' in question_lower or 'sous-ajustement' in question_lower:
            if 'evaluation' in context and context['evaluation'] is not None:
                underfitting = context['evaluation'].get('underfitting', {})
                if underfitting and underfitting.get('detected'):
                    return "Oui, de l'underfitting a été détecté. Le modèle ne performe pas bien sur les données d'entraînement."
                return "Aucun underfitting significatif détecté."
            return "Je n'ai pas encore évalué l'underfitting."
        
        # Réponse par défaut
        return "Je peux vous aider avec les données, les modèles, les performances, le déséquilibre, l'overfitting/underfitting, et les pistes d'amélioration. Posez-moi une question plus spécifique !"

