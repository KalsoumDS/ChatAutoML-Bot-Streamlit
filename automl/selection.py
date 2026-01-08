"""
Module de sélection du meilleur modèle
"""
import pandas as pd
from typing import Dict, List, Any, Optional


class ModelSelector:
    """Sélectionne le meilleur modèle selon la métrique principale"""
    
    def __init__(self, task_type: str, metric: str = 'auto'):
        """
        Args:
            task_type: 'classification' ou 'regression'
            metric: Métrique principale pour la sélection
        """
        self.task_type = task_type
        self.metric = metric
        self._determine_metric_key()
    
    def _determine_metric_key(self):
        """Détermine la clé de métrique à utiliser"""
        if self.metric == 'auto':
            if self.task_type == 'classification':
                self.metric_key = 'accuracy'
            else:
                self.metric_key = 'r2'
        else:
            metric_mapping = {
                'accuracy': 'accuracy',
                'f1': 'f1_macro',
                'f1_macro': 'f1_macro',
                'f1_weighted': 'f1_weighted',
                'auc': 'roc_auc',
                'roc_auc': 'roc_auc',
                'mse': 'neg_mse',
                'rmse': 'neg_rmse',
                'r2': 'r2',
                'r²': 'r2'
            }
            self.metric_key = metric_mapping.get(self.metric.lower(), self.metric)
    
    def select_best_model(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sélectionne le meilleur modèle parmi les résultats de recherche
        
        Args:
            search_results: Liste de résultats de ModelSearcher
            
        Returns:
            Dictionnaire avec le meilleur modèle et un résumé comparatif
        """
        if not search_results:
            raise ValueError("Aucun résultat de recherche disponible")
        
        # Extraire les scores selon la métrique
        model_scores = []
        for result in search_results:
            scores = result.get('scores', {})
            score = scores.get(self.metric_key, result.get('best_score', 0))
            
            # Pour les métriques négatives (MSE, RMSE), on veut le moins négatif
            if 'neg' in self.metric_key:
                score = -score  # Inverser pour comparaison
            
            model_scores.append({
                'model_name': result['model_name'],
                'score': score,
                'result': result
            })
        
        # Trier par score (décroissant pour accuracy/f1/auc/r2, croissant pour mse/rmse)
        reverse = self.metric_key not in ['neg_mse', 'neg_rmse', 'mse', 'rmse']
        model_scores.sort(key=lambda x: x['score'], reverse=reverse)
        
        best_result = model_scores[0]['result']
        
        # Créer un résumé comparatif
        summary = self._create_comparison_summary(model_scores)
        
        return {
            'best_model': best_result['model'],
            'best_model_name': best_result['model_name'],
            'best_params': best_result['best_params'],
            'best_score': best_result['best_score'],
            'all_scores': best_result['scores'],
            'comparison_summary': summary
        }
    
    def _create_comparison_summary(self, model_scores: List[Dict]) -> pd.DataFrame:
        """Crée un DataFrame comparatif de tous les modèles"""
        rows = []
        for item in model_scores:
            result = item['result']
            scores = result.get('scores', {})
            
            row = {
                'Model': result['model_name'],
                'Best Score': round(item['score'], 4),
                'Time (s)': round(result['elapsed_time'], 2)
            }
            
            # Ajouter les autres métriques
            if self.task_type == 'classification':
                row['Accuracy'] = round(scores.get('accuracy', 0), 4)
                row['F1 Macro'] = round(scores.get('f1_macro', 0), 4)
                row['F1 Weighted'] = round(scores.get('f1_weighted', 0), 4)
                if 'roc_auc' in scores:
                    row['ROC AUC'] = round(scores.get('roc_auc', 0), 4)
            else:
                row['R²'] = round(scores.get('r2', 0), 4)
                row['RMSE'] = round(scores.get('neg_rmse', 0), 4) if 'neg_rmse' in scores else None
                row['MSE'] = round(scores.get('neg_mse', 0), 4) if 'neg_mse' in scores else None
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df


