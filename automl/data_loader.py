"""
Module de chargement et d'analyse initiale des données tabulaires
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class DataLoader:
    """Charge et analyse les données tabulaires depuis différents formats"""
    
    SUPPORTED_FORMATS = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.parquet': pd.read_parquet,
        '.json': None  # Géré séparément
    }
    
    def __init__(self, max_size_mb: int = 100):
        """
        Args:
            max_size_mb: Taille maximale du fichier en Mo
        """
        self.max_size_mb = max_size_mb
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Charge un fichier dans un DataFrame pandas
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            DataFrame pandas
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Format non supporté: {extension}. Formats supportés: {list(self.SUPPORTED_FORMATS.keys())}")
        
        # Vérifier la taille du fichier
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            raise ValueError(f"Fichier trop volumineux ({file_size_mb:.2f} Mo). Maximum: {self.max_size_mb} Mo")
        
        # Charger selon le format
        if extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("JSON doit être une liste de dictionnaires")
        else:
            read_func = self.SUPPORTED_FORMATS[extension]
            df = read_func(file_path)
        
        if df.empty:
            raise ValueError("Le fichier est vide")
        
        return df
    
    def analyze_dataset(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict:
        """
        Génère un résumé automatique du dataset
        
        Args:
            df: DataFrame à analyser
            target_column: Nom de la colonne cible (optionnel)
            
        Returns:
            Dictionnaire contenant les statistiques
        """
        analysis = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'columns': {},
            'missing_values': {},
            'target_info': None
        }
        
        # Analyser chaque colonne
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'type': self._detect_column_type(df[col]),
                'unique_count': df[col].nunique(),
                'missing_count': df[col].isna().sum(),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100
            }
            
            if col_info['type'] == 'numeric':
                col_info['mean'] = float(df[col].mean()) if df[col].dtype in ['int64', 'float64'] else None
                col_info['std'] = float(df[col].std()) if df[col].dtype in ['int64', 'float64'] else None
                col_info['min'] = float(df[col].min()) if df[col].dtype in ['int64', 'float64'] else None
                col_info['max'] = float(df[col].max()) if df[col].dtype in ['int64', 'float64'] else None
            elif col_info['type'] == 'categorical':
                col_info['top_values'] = df[col].value_counts().head(5).to_dict()
            
            analysis['columns'][col] = col_info
            analysis['missing_values'][col] = {
                'count': int(col_info['missing_count']),
                'percentage': round(col_info['missing_percentage'], 2)
            }
        
        # Analyser la colonne cible si spécifiée
        if target_column and target_column in df.columns:
            target_series = df[target_column]
            analysis['target_info'] = {
                'column': target_column,
                'type': self._detect_column_type(target_series),
                'unique_count': target_series.nunique(),
                'distribution': target_series.value_counts().to_dict(),
                'is_balanced': self._check_balance(target_series)
            }
        
        return analysis
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Détecte le type d'une colonne"""
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 2:
                return 'boolean'
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        else:
            return 'categorical'
    
    def _check_balance(self, series: pd.Series) -> bool:
        """Vérifie si une série est équilibrée (pour classification)"""
        if series.nunique() <= 1:
            return True
        
        value_counts = series.value_counts()
        max_proportion = value_counts.max() / len(series)
        return max_proportion < 0.7  # Considéré déséquilibré si une classe > 70%
    
    def detect_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        """
        Détecte automatiquement le type de tâche (classification ou régression)
        
        Args:
            df: DataFrame
            target_column: Nom de la colonne cible
            
        Returns:
            'classification' ou 'regression'
        """
        target = df[target_column]
        
        # Si non numérique, c'est de la classification
        if not pd.api.types.is_numeric_dtype(target):
            return 'classification'
        
        # Si numérique avec peu de valeurs distinctes, classification
        unique_ratio = target.nunique() / len(target)
        if unique_ratio < 0.05 or target.nunique() <= 10:
            return 'classification'
        
        # Sinon, régression
        return 'regression'


