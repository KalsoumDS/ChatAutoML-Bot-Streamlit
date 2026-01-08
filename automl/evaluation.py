"""
Module d'évaluation des modèles et détection d'overfitting/underfitting
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from typing import Dict, List, Tuple, Optional, Any


class ModelEvaluator:
    """Évalue les modèles et détecte overfitting/underfitting"""
    
    def __init__(self, task_type: str):
        """
        Args:
            task_type: 'classification' ou 'regression'
        """
        self.task_type = task_type
    
    def evaluate_model(self, model, X_train: np.ndarray, y_train: pd.Series,
                      X_test: np.ndarray, y_test: pd.Series,
                      metric: str = 'auto') -> Dict[str, Any]:
        """
        Évalue un modèle sur train et test
        
        Args:
            model: Modèle entraîné
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de test
            metric: Métrique principale
            
        Returns:
            Dictionnaire avec toutes les métriques et diagnostics
        """
        # Prédictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilités pour classification
        if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                y_train_proba = model.predict_proba(X_train)
                y_test_proba = model.predict_proba(X_test)
            except:
                y_train_proba = None
                y_test_proba = None
        else:
            y_train_proba = None
            y_test_proba = None
        
        # Calculer les métriques
        train_metrics = self._compute_metrics(y_train, y_train_pred, y_train_proba, 'train')
        test_metrics = self._compute_metrics(y_test, y_test_pred, y_test_proba, 'test')
        
        # Détecter overfitting/underfitting
        overfitting_info = self._detect_overfitting(train_metrics, test_metrics, metric)
        underfitting_info = self._detect_underfitting(train_metrics, test_metrics, metric)
        
        # Rapport de classification si classification
        classification_report_dict = None
        confusion_matrix_array = None
        if self.task_type == 'classification':
            try:
                classification_report_dict = classification_report(
                    y_test, y_test_pred, output_dict=True
                )
                confusion_matrix_array = confusion_matrix(y_test, y_test_pred).tolist()
            except:
                pass
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'overfitting': overfitting_info,
            'underfitting': underfitting_info,
            'classification_report': classification_report_dict,
            'confusion_matrix': confusion_matrix_array
        }
    
    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                        y_proba: Optional[np.ndarray], split: str) -> Dict[str, float]:
        """Calcule les métriques pour un split"""
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            # F1 score
            try:
                metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
                metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted'))
            except:
                pass
            
            # AUC (si binaire et probabilités disponibles)
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                    else:
                        # Multi-class AUC
                        metrics['roc_auc'] = float(roc_auc_score(
                            y_true, y_proba, multi_class='ovr', average='macro'
                        ))
                except:
                    pass
        
        else:  # Regression
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
        
        return metrics
    
    def _detect_overfitting(self, train_metrics: Dict, test_metrics: Dict,
                           metric: str) -> Dict[str, Any]:
        """Détecte l'overfitting"""
        if metric == 'auto':
            if self.task_type == 'classification':
                metric_key = 'accuracy'
            else:
                metric_key = 'r2'
        else:
            metric_mapping = {
                'accuracy': 'accuracy',
                'f1': 'f1_macro',
                'auc': 'roc_auc',
                'mse': 'mse',
                'rmse': 'rmse',
                'r2': 'r2',
                'r²': 'r2'
            }
            metric_key = metric_mapping.get(metric.lower(), metric)
        
        train_score = train_metrics.get(metric_key)
        test_score = test_metrics.get(metric_key)
        
        if train_score is None or test_score is None:
            return {'detected': False, 'reason': 'Métrique non disponible'}
        
        # Pour les métriques où plus c'est mieux (accuracy, f1, auc, r2)
        if metric_key in ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc', 'r2']:
            gap = train_score - test_score
            threshold = 0.1  # 10% de différence
            if self.task_type == 'classification' and metric_key == 'accuracy':
                threshold = 0.05  # 5% pour accuracy
        else:  # Pour MSE, RMSE (moins c'est mieux)
            gap = test_score - train_score
            threshold = 0.1
        
        is_overfitting = gap > threshold and train_score > 0.7  # Overfitting si gap important ET train score bon
        
        return {
            'detected': is_overfitting,
            'train_score': train_score,
            'test_score': test_score,
            'gap': gap,
            'threshold': threshold
        }
    
    def _detect_underfitting(self, train_metrics: Dict, test_metrics: Dict,
                            metric: str) -> Dict[str, Any]:
        """Détecte l'underfitting"""
        if metric == 'auto':
            if self.task_type == 'classification':
                metric_key = 'accuracy'
            else:
                metric_key = 'r2'
        else:
            metric_mapping = {
                'accuracy': 'accuracy',
                'f1': 'f1_macro',
                'auc': 'roc_auc',
                'mse': 'mse',
                'rmse': 'rmse',
                'r2': 'r2',
                'r²': 'r2'
            }
            metric_key = metric_mapping.get(metric.lower(), metric)
        
        train_score = train_metrics.get(metric_key)
        test_score = test_metrics.get(metric_key)
        
        if train_score is None or test_score is None:
            return {'detected': False, 'reason': 'Métrique non disponible'}
        
        # Underfitting si les scores sont faibles sur train ET test
        if metric_key in ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc', 'r2']:
            threshold = 0.5  # Score faible si < 0.5
            is_underfitting = train_score < threshold and test_score < threshold
        else:  # Pour MSE, RMSE
            # Difficile de définir un seuil absolu, on regarde plutôt si les deux sont élevés
            is_underfitting = False
        
        return {
            'detected': is_underfitting,
            'train_score': train_score,
            'test_score': test_score
        }
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Extrait l'importance des features si disponible
        
        Returns:
            Dictionnaire {feature: importance} ou None
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances.tolist()))
            elif hasattr(model, 'coef_'):
                # Pour les modèles linéaires, utiliser les coefficients absolus
                coef = np.abs(model.coef_)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                return dict(zip(feature_names, coef.tolist()))
        except:
            pass
        
        return None

