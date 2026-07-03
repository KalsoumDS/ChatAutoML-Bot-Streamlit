"""
Module de recherche de modèles et hyperparamètres (AutoML)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, List, Any, Optional
from .model_zoo import ModelZoo
import time


class ModelSearcher:
    """Effectue la recherche de modèles et hyperparamètres"""
    
    def __init__(self, cv: int = 5, scoring: Optional[str] = None, 
                 search_method: str = 'grid', n_iter: int = 20):
        """
        Args:
            cv: Nombre de folds pour la validation croisée
            scoring: Métrique principale (None = métrique par défaut)
            search_method: 'grid' ou 'random'
            n_iter: Nombre d'itérations pour RandomizedSearchCV
        """
        self.cv = cv
        self.scoring = scoring
        self.search_method = search_method
        self.n_iter = n_iter
        self.results = []
    
    def search_models(self, X_train: np.ndarray, y_train: pd.Series,
                     task_type: str, metric: str = 'auto') -> List[Dict[str, Any]]:
        """
        Recherche les meilleurs modèles et hyperparamètres
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            task_type: 'classification' ou 'regression'
            metric: Métrique principale
            
        Returns:
            Liste de dictionnaires avec les résultats de chaque modèle
        """
        # Déterminer la métrique si auto
        if metric == 'auto':
            if task_type == 'classification':
                metric = 'accuracy'
            else:
                metric = 'neg_mean_squared_error'
        
        # Mapper les métriques utilisateur vers sklearn
        metric_mapping = {
            'accuracy': 'accuracy',
            'f1': 'f1_macro',
            'f1_macro': 'f1_macro',
            'f1_micro': 'f1_micro',
            'f1_weighted': 'f1_weighted',
            'auc': 'roc_auc',
            'roc_auc': 'roc_auc',
            'mse': 'neg_mean_squared_error',
            'rmse': 'neg_root_mean_squared_error',
            'r2': 'r2',
            'r²': 'r2'
        }
        
        scoring = metric_mapping.get(metric.lower(), metric)
        
        # Obtenir les modèles
        models = ModelZoo.get_models_for_task(task_type)
        
        results = []
        
        for model_name, model_config in models.items():
            print(f"Recherche pour {model_name}...")
            start_time = time.time()
            
            try:
                model_class = model_config['model']
                param_grid = model_config['params']
                
                # Créer le modèle de base
                base_model = model_class()
                
                # Recherche d'hyperparamètres
                if self.search_method == 'grid' and len(param_grid) > 0:
                    search = GridSearchCV(
                        base_model,
                        param_grid,
                        cv=min(self.cv, 3),  # max 3 folds sur Streamlit Cloud
                        scoring=scoring,
                        n_jobs=1,  # pas de parallélisme sur Streamlit Cloud
                        verbose=0
                    )
                elif self.search_method == 'random' and len(param_grid) > 0:
                    search = RandomizedSearchCV(
                        base_model,
                        param_grid,
                        cv=min(self.cv, 3),
                        scoring=scoring,
                        n_iter=min(self.n_iter, 10),  # max 10 itérations
                        n_jobs=1,
                        verbose=0,
                        random_state=42
                    )
                else:
                    # Pas de recherche, utiliser les paramètres par défaut
                    search = base_model
                
                # Entraîner
                if hasattr(search, 'fit'):
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    best_score = search.best_score_
                    
                    # Calculer les scores sur toutes les métriques
                    scores = self._compute_all_scores(best_model, X_train, y_train, task_type)
                    # Utiliser le score de la métrique principale depuis scores si disponible
                    metric_key_for_score = metric_mapping.get(metric.lower(), scoring)
                    if metric_key_for_score in scores:
                        best_score = scores[metric_key_for_score]
                else:
                    search.fit(X_train, y_train)
                    best_model = search
                    best_params = {}
                    scores = self._compute_all_scores(best_model, X_train, y_train, task_type)
                    # Utiliser la clé de métrique appropriée
                    default_key = 'accuracy' if task_type == 'classification' else 'r2'
                    metric_key = metric_mapping.get(metric.lower(), default_key)
                    # Pour les métriques négatives, utiliser la clé correspondante
                    if metric_key.startswith('neg_'):
                        score_key = metric_key.replace('neg_', 'neg_')
                    else:
                        score_key = metric_key
                    best_score = scores.get(score_key, scores.get(default_key, 0))
                
                elapsed_time = time.time() - start_time
                
                result = {
                    'model_name': model_name,
                    'model': best_model,
                    'best_params': best_params,
                    'best_score': float(best_score),
                    'scores': scores,
                    'elapsed_time': elapsed_time
                }
                
                results.append(result)
                print(f"  ✓ {model_name}: score={best_score:.4f} (temps: {elapsed_time:.2f}s)")
                
            except Exception as e:
                print(f"  ✗ {model_name} a échoué: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def _compute_all_scores(self, model, X: np.ndarray, y: pd.Series, 
                           task_type: str) -> Dict[str, float]:
        """Calcule toutes les métriques pertinentes"""
        from sklearn.model_selection import cross_val_score
        
        scores = {}
        
        if task_type == 'classification':
            metrics = {
                'accuracy': 'accuracy',
                'f1_macro': 'f1_macro',
                'f1_weighted': 'f1_weighted',
                'roc_auc': 'roc_auc'
            }
        else:
            metrics = {
                'neg_mse': 'neg_mean_squared_error',
                'neg_rmse': 'neg_root_mean_squared_error',
                'r2': 'r2'
            }
        
        for score_name, sklearn_metric in metrics.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=self.cv, scoring=sklearn_metric)
                scores[score_name] = float(cv_scores.mean())
                scores[f'{score_name}_std'] = float(cv_scores.std())
            except:
                pass
        
        return scores

