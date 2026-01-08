"""
Création alternative du dataset California Housing
"""
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from sklearn.datasets import fetch_california_housing
    
    print("Téléchargement de California Housing depuis sklearn...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    df['MedHouseVal'] = housing.target
    
    file_path = Path(__file__).parent / "california_housing.csv"
    df.to_csv(file_path, index=False)
    print(f"✓ California Housing sauvegardé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
except Exception as e:
    print(f"Erreur : {e}")
    print("Création d'un dataset similaire...")
    
    # Créer un dataset similaire
    np.random.seed(42)
    n_samples = 20640
    
    # Features similaires à California Housing
    df = pd.DataFrame({
        'MedInc': np.random.lognormal(3.5, 0.5, n_samples),
        'HouseAge': np.random.randint(1, 53, n_samples),
        'AveRooms': np.random.lognormal(1.0, 0.5, n_samples) + 1,
        'AveBedrms': np.random.lognormal(0.0, 0.5, n_samples) + 0.5,
        'Population': np.random.randint(1, 4000, n_samples),
        'AveOccup': np.random.lognormal(0.5, 0.5, n_samples) + 1,
        'Latitude': np.random.uniform(32.5, 42.0, n_samples),
        'Longitude': np.random.uniform(-124.0, -114.0, n_samples),
    })
    
    # Target basé sur les features
    df['target'] = (df['MedInc'] * 0.5 + 
                    df['AveRooms'] * 0.3 + 
                    np.random.randn(n_samples) * 0.5)
    df['MedHouseVal'] = df['target']
    
    file_path = Path(__file__).parent / "california_housing.csv"
    df.to_csv(file_path, index=False)
    print(f"✓ Dataset similaire créé : {df.shape[0]} lignes, {df.shape[1]} colonnes")


