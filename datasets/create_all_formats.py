"""
Script pour créer des datasets dans tous les formats supportés
CSV, Excel (.xlsx, .xls), Parquet, JSON
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json


def create_all_formats():
    """Crée des datasets dans tous les formats supportés"""
    print("=" * 60)
    print("Création de datasets dans tous les formats")
    print("=" * 60)
    
    datasets_dir = Path(__file__).parent
    
    # 1. Classification binaire équilibrée
    print("\n📊 Création : Classification binaire équilibrée")
    np.random.seed(42)
    n_samples = 200
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    X2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    df_binary = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df_binary['target'] = y
    df_binary['target'] = df_binary['target'].map({0: 'class_A', 1: 'class_B'})
    
    save_all_formats(df_binary, 'binary_classification', datasets_dir)
    
    # 2. Classification multiclasse
    print("\n📊 Création : Classification multiclasse")
    np.random.seed(42)
    n_samples = 150
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    X2 = np.random.randn(n_samples, 2) + np.array([-2, 2])
    X3 = np.random.randn(n_samples, 2) + np.array([0, -2])
    X = np.vstack([X1, X2, X3])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples), np.full(n_samples, 2)])
    
    df_multiclass = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df_multiclass['target'] = y
    df_multiclass['target'] = df_multiclass['target'].map({0: 'class_A', 1: 'class_B', 2: 'class_C'})
    
    save_all_formats(df_multiclass, 'multiclass_classification', datasets_dir)
    
    # 3. Régression
    print("\n📊 Création : Régression")
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    
    df_regression = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    df_regression['target'] = y
    
    save_all_formats(df_regression, 'regression', datasets_dir)
    
    # 4. Avec valeurs manquantes
    print("\n📊 Création : Avec valeurs manquantes")
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 4)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.3
    
    df_missing = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    df_missing['target'] = y
    
    # Ajouter des valeurs manquantes
    missing_indices = np.random.choice(df_missing.index, size=int(n_samples * 0.1), replace=False)
    df_missing.loc[missing_indices, 'feature_2'] = np.nan
    
    missing_indices = np.random.choice(df_missing.index, size=int(n_samples * 0.05), replace=False)
    df_missing.loc[missing_indices, 'feature_3'] = np.nan
    
    save_all_formats(df_missing, 'with_missing_values', datasets_dir)
    
    # 5. Features mixtes (numériques + catégorielles)
    print("\n📊 Création : Features mixtes")
    np.random.seed(42)
    n_samples = 250
    X_num = np.random.randn(n_samples, 2)
    categories = ['A', 'B', 'C']
    X_cat = np.random.choice(categories, size=n_samples)
    y = X_num[:, 0] + (X_cat == 'A').astype(int) * 2 + np.random.randn(n_samples) * 0.5
    
    df_mixed = pd.DataFrame(X_num, columns=['numeric_1', 'numeric_2'])
    df_mixed['categorical'] = X_cat
    df_mixed['target'] = y
    
    save_all_formats(df_mixed, 'mixed_features', datasets_dir)
    
    # 6. Classification déséquilibrée
    print("\n📊 Création : Classification déséquilibrée")
    np.random.seed(42)
    n_samples_majority = 800
    n_samples_minority = 50
    X1 = np.random.randn(n_samples_majority, 3) + np.array([0, 0, 0])
    X2 = np.random.randn(n_samples_minority, 3) + np.array([3, 3, 3])
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples_majority), np.ones(n_samples_minority)])
    
    df_imbalanced = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    df_imbalanced['target'] = y
    df_imbalanced['target'] = df_imbalanced['target'].map({0: 'majority', 1: 'minority'})
    
    save_all_formats(df_imbalanced, 'imbalanced_binary', datasets_dir)
    
    # 7. Dataset petit (pour tests rapides)
    print("\n📊 Création : Dataset petit")
    np.random.seed(42)
    n_samples = 50
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] > 0).astype(int)
    
    df_small = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df_small['target'] = y
    df_small['target'] = df_small['target'].map({0: 'negative', 1: 'positive'})
    
    save_all_formats(df_small, 'small', datasets_dir)
    
    # 8. Charger les datasets réels existants et les convertir
    print("\n📊 Conversion des datasets réels existants")
    
    # Iris
    if (datasets_dir / 'iris.csv').exists():
        df_iris = pd.read_csv(datasets_dir / 'iris.csv')
        save_all_formats(df_iris, 'iris', datasets_dir)
    
    # Titanic
    if (datasets_dir / 'titanic.csv').exists():
        df_titanic = pd.read_csv(datasets_dir / 'titanic.csv')
        # Limiter à 1000 lignes pour Excel
        if len(df_titanic) > 1000:
            df_titanic_sample = df_titanic.head(1000)
            save_all_formats(df_titanic_sample, 'titanic_sample', datasets_dir)
        else:
            save_all_formats(df_titanic, 'titanic', datasets_dir)
    
    print("\n" + "=" * 60)
    print("✅ Tous les formats ont été créés avec succès !")
    print("=" * 60)
    print("\nFormats disponibles pour chaque dataset :")
    print("  - CSV (.csv)")
    print("  - Excel (.xlsx)")
    print("  - Parquet (.parquet)")
    print("  - JSON (.json)")
    print("\n📁 Tous les fichiers sont dans le dossier 'datasets/'")


def save_all_formats(df: pd.DataFrame, name: str, datasets_dir: Path):
    """Sauvegarde un DataFrame dans tous les formats supportés"""
    
    # 1. CSV
    csv_path = datasets_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV créé : {csv_path.name}")
    
    # 2. Excel (.xlsx)
    try:
        xlsx_path = datasets_dir / f"{name}.xlsx"
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
        print(f"  ✓ Excel (.xlsx) créé : {xlsx_path.name}")
    except Exception as e:
        print(f"  ✗ Excel (.xlsx) : {e}")
    
    # 3. Excel (.xls) - format ancien
    try:
        xls_path = datasets_dir / f"{name}.xls"
        # Note: .xls nécessite xlwt, mais on peut créer un .xlsx et le renommer
        # Pour simplifier, on skip .xls car c'est un format obsolète
        print(f"  ⚠ Excel (.xls) : Format obsolète, utilisez .xlsx")
    except Exception as e:
        pass
    
    # 4. Parquet
    try:
        parquet_path = datasets_dir / f"{name}.parquet"
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        print(f"  ✓ Parquet créé : {parquet_path.name}")
    except Exception as e:
        print(f"  ✗ Parquet : {e}")
    
    # 5. JSON tabulaire (liste de dictionnaires)
    try:
        json_path = datasets_dir / f"{name}.json"
        # Convertir en liste de dictionnaires
        json_data = df.to_dict('records')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✓ JSON créé : {json_path.name}")
    except Exception as e:
        print(f"  ✗ JSON : {e}")


if __name__ == "__main__":
    create_all_formats()


