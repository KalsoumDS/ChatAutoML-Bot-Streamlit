"""
Module de gestion du déséquilibre des classes (classification)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


class Resampler:
    """Gère le rééchantillonnage pour équilibrer les classes"""
    
    def __init__(self, strategy: str = 'auto'):
        """
        Args:
            strategy: 'auto', 'smote', 'oversample', 'undersample', 'smote_tomek', ou None
        """
        self.strategy = strategy
        self.resampler = None
        self.applied = False
    
    def check_imbalance(self, y: pd.Series, threshold: float = 0.7) -> Tuple[bool, Dict]:
        """
        Vérifie si les classes sont déséquilibrées
        
        Args:
            y: Série des labels
            threshold: Seuil de déséquilibre (proportion de la classe majoritaire)
            
        Returns:
            Tuple (is_imbalanced, distribution_info)
        """
        value_counts = y.value_counts()
        max_proportion = value_counts.max() / len(y)
        
        distribution = {
            'class_counts': value_counts.to_dict(),
            'class_proportions': (value_counts / len(y)).to_dict(),
            'max_proportion': max_proportion,
            'is_imbalanced': max_proportion > threshold
        }
        
        return distribution['is_imbalanced'], distribution
    
    def apply_resampling(self, X: np.ndarray, y: pd.Series, 
                        strategy: Optional[str] = None) -> Tuple[np.ndarray, pd.Series]:
        """
        Applique le rééchantillonnage
        
        Args:
            X: Features
            y: Labels
            strategy: Stratégie à utiliser (override self.strategy)
            
        Returns:
            Tuple (X_resampled, y_resampled)
        """
        strategy = strategy or self.strategy
        
        if strategy is None or strategy == 'none':
            self.applied = False
            return X, y
        
        # Vérifier le déséquilibre
        is_imbalanced, dist_info = self.check_imbalance(y)
        
        if not is_imbalanced:
            self.applied = False
            return X, y
        
        # Choisir la stratégie automatiquement si 'auto'
        if strategy == 'auto':
            if y.nunique() == 2:  # Classification binaire
                strategy = 'smote'
            else:  # Classification multiclasse
                strategy = 'smote'
        
        # Appliquer la stratégie
        if strategy == 'smote':
            try:
                self.resampler = SMOTE(random_state=42, k_neighbors=min(5, y.value_counts().min() - 1))
                X_resampled, y_resampled = self.resampler.fit_resample(X, y)
            except Exception as e:
                # Si SMOTE échoue (pas assez de voisins), utiliser oversampling simple
                print(f"SMOTE échoué: {e}. Utilisation de l'oversampling simple.")
                from imblearn.over_sampling import RandomOverSampler
                self.resampler = RandomOverSampler(random_state=42)
                X_resampled, y_resampled = self.resampler.fit_resample(X, y)
        
        elif strategy == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            self.resampler = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = self.resampler.fit_resample(X, y)
        
        elif strategy == 'undersample':
            self.resampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = self.resampler.fit_resample(X, y)
        
        elif strategy == 'smote_tomek':
            self.resampler = SMOTETomek(random_state=42)
            X_resampled, y_resampled = self.resampler.fit_resample(X, y)
        
        else:
            self.applied = False
            return X, y
        
        self.applied = True
        
        # Convertir y_resampled en Series si nécessaire
        if isinstance(y_resampled, np.ndarray):
            y_resampled = pd.Series(y_resampled)
        
        return X_resampled, y_resampled
    
    def get_strategy_info(self) -> Dict:
        """Retourne des informations sur la stratégie appliquée"""
        return {
            'applied': self.applied,
            'strategy': self.strategy if self.applied else None
        }

