"""
Module de prétraitement des données tabulaires
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict


class Preprocessor:
    """Gère le prétraitement des données tabulaires"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            test_size: Proportion des données de test
            random_state: Seed pour la reproductibilité
        """
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sépare les features et la cible
        
        Args:
            df: DataFrame complet
            target_column: Nom de la colonne cible
            
        Returns:
            Tuple (X, y)
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    
    def identify_column_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identifie les types de colonnes
        
        Returns:
            Dictionnaire avec 'numeric', 'categorical', 'boolean'
        """
        column_types = {
            'numeric': [],
            'categorical': [],
            'boolean': []
        }
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                if X[col].nunique() <= 2:
                    column_types['boolean'].append(col)
                else:
                    column_types['numeric'].append(col)
            else:
                column_types['categorical'].append(col)
        
        return column_types
    
    def create_preprocessor(self, X: pd.DataFrame, scaling: bool = True) -> ColumnTransformer:
        """
        Crée un ColumnTransformer pour le prétraitement
        
        Args:
            X: DataFrame des features
            scaling: Si True, normalise les variables numériques
            
        Returns:
            ColumnTransformer configuré
        """
        column_types = self.identify_column_types(X)
        
        transformers = []
        
        # Traitement des colonnes numériques
        if column_types['numeric']:
            numeric_steps = [
                ('imputer', SimpleImputer(strategy='median')),
            ]
            if scaling:
                numeric_steps.append(('scaler', StandardScaler()))
            
            numeric_transformer = Pipeline(numeric_steps)
            transformers.append(
                ('num', numeric_transformer, column_types['numeric'])
            )
        
        # Traitement des colonnes catégorielles
        if column_types['categorical']:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(
                ('cat', categorical_transformer, column_types['categorical'])
            )
        
        # Traitement des colonnes booléennes (converties en numériques)
        if column_types['boolean']:
            boolean_steps = [
                ('imputer', SimpleImputer(strategy='most_frequent')),
            ]
            if scaling:
                boolean_steps.append(('scaler', StandardScaler()))
            
            boolean_transformer = Pipeline(boolean_steps)
            transformers.append(
                ('bool', boolean_transformer, column_types['boolean'])
            )
        
        if not transformers:
            raise ValueError("Aucune colonne à traiter")
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit et transform les données"""
        if self.preprocessor is None:
            self.create_preprocessor(X)
        
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Récupérer les noms de features
        try:
            self.feature_names = self.preprocessor.get_feature_names_out()
        except:
            self.feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform les données (après fit)"""
        if self.preprocessor is None:
            raise ValueError("Le preprocessor doit être fit d'abord")
        
        return self.preprocessor.transform(X)
    
    def split_data(self, X: np.ndarray, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Divise les données en train et test
        
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if y.dtype == 'object' or y.nunique() < 20 else None
        )

