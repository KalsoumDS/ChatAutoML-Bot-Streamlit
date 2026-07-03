"""
Module définissant les modèles et grilles d'hyperparamètres pour AutoML
"""
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from typing import Dict, List, Any
import numpy as np


class ModelZoo:
    """Définit les modèles et leurs grilles d'hyperparamètres"""
    
    @staticmethod
    def get_classification_models() -> Dict[str, Dict[str, Any]]:
        """
        Retourne les modèles de classification et leurs grilles
        
        Returns:
            Dictionnaire {nom_modele: {'model': classe, 'params': grille}}
        """
        models = {
            'LogisticRegression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear'],
                    'max_iter': [1000],
                    'random_state': [42]
                }
            },
            'RandomForestClassifier': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10],
                    'random_state': [42]
                }
            },
            'GradientBoostingClassifier': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'random_state': [42]
                }
            },
            'SVC': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf'],
                    'probability': [True],
                    'random_state': [42]
                }
            },
            'KNeighborsClassifier': {
                'model': KNeighborsClassifier,
                'params': {
                    'n_neighbors': [3, 5, 9],
                    'weights': ['uniform', 'distance'],
                }
            }
        }
        
        # Ajouter XGBoost si disponible
        try:
            import xgboost as xgb
            models['XGBoostClassifier'] = {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'random_state': [42]
                }
            }
        except ImportError:
            pass
        
        # Ajouter LightGBM si disponible
        try:
            import lightgbm as lgb
            models['LightGBMClassifier'] = {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'random_state': [42],
                    'verbosity': [-1]
                }
            }
        except ImportError:
            pass
        
        return models
    
    @staticmethod
    def get_regression_models() -> Dict[str, Dict[str, Any]]:
        """
        Retourne les modèles de régression et leurs grilles
        
        Returns:
            Dictionnaire {nom_modele: {'model': classe, 'params': grille}}
        """
        models = {
            'LinearRegression': {
                'model': LinearRegression,
                'params': {}
            },
            'Ridge': {
                'model': Ridge,
                'params': {
                    'alpha': [0.1, 1, 10],
                    'random_state': [42]
                }
            },
            'Lasso': {
                'model': Lasso,
                'params': {
                    'alpha': [0.1, 1, 10],
                    'max_iter': [1000],
                    'random_state': [42]
                }
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10],
                    'random_state': [42]
                }
            },
            'GradientBoostingRegressor': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'random_state': [42]
                }
            },
            'KNeighborsRegressor': {
                'model': KNeighborsRegressor,
                'params': {
                    'n_neighbors': [3, 5, 9],
                    'weights': ['uniform', 'distance'],
                }
            }
        }
        
        # Ajouter XGBoost si disponible
        try:
            import xgboost as xgb
            models['XGBoostRegressor'] = {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'random_state': [42]
                }
            }
        except ImportError:
            pass
        
        # Ajouter LightGBM si disponible
        try:
            import lightgbm as lgb
            models['LightGBMRegressor'] = {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'random_state': [42],
                    'verbosity': [-1]
                }
            }
        except ImportError:
            pass
        
        return models
    
    @staticmethod
    def get_models_for_task(task_type: str) -> Dict[str, Dict[str, Any]]:
        """Retourne les modèles selon le type de tâche"""
        if task_type == 'classification':
            return ModelZoo.get_classification_models()
        elif task_type == 'regression':
            return ModelZoo.get_regression_models()
        else:
            raise ValueError(f"Type de tâche inconnu: {task_type}")


