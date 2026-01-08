"""
Script pour télécharger les jeux de données de référence
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing, load_iris
import urllib.request
import zipfile
import shutil


def download_file(url: str, destination: str):
    """Télécharge un fichier depuis une URL"""
    print(f"Téléchargement de {url}...")
    urllib.request.urlretrieve(url, destination)
    print(f"✓ Téléchargé : {destination}")


def download_titanic():
    """Télécharge le dataset Titanic depuis Kaggle"""
    print("\n=== Dataset Titanic ===")
    try:
        # Essayer de télécharger depuis une source publique
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        file_path = Path(__file__).parent / "titanic.csv"
        download_file(url, str(file_path))
        
        # Vérifier et nettoyer
        df = pd.read_csv(file_path)
        print(f"✓ Titanic chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return True
    except Exception as e:
        print(f"✗ Erreur : {e}")
        print("Note : Vous pouvez télécharger manuellement depuis https://www.kaggle.com/c/titanic/data")
        return False


def download_iris():
    """Télécharge le dataset Iris depuis sklearn"""
    print("\n=== Dataset Iris ===")
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        file_path = Path(__file__).parent / "iris.csv"
        df.to_csv(file_path, index=False)
        print(f"✓ Iris sauvegardé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return True
    except Exception as e:
        print(f"✗ Erreur : {e}")
        return False


def download_california_housing():
    """Télécharge le dataset California Housing depuis sklearn"""
    print("\n=== Dataset California Housing ===")
    try:
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['target'] = housing.target
        df['MedHouseVal'] = housing.target  # Nom plus explicite
        
        file_path = Path(__file__).parent / "california_housing.csv"
        df.to_csv(file_path, index=False)
        print(f"✓ California Housing sauvegardé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return True
    except Exception as e:
        print(f"✗ Erreur : {e}")
        return False


def download_wine_quality():
    """Télécharge le dataset Wine Quality depuis UCI"""
    print("\n=== Dataset Wine Quality ===")
    try:
        # Dataset Wine Quality (version rouge)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        file_path = Path(__file__).parent / "wine_quality_red.csv"
        download_file(url, str(file_path))
        
        df = pd.read_csv(file_path, sep=';')
        print(f"✓ Wine Quality (rouge) chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Dataset Wine Quality (version blanc)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        file_path = Path(__file__).parent / "wine_quality_white.csv"
        download_file(url, str(file_path))
        
        df = pd.read_csv(file_path, sep=';')
        print(f"✓ Wine Quality (blanc) chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        return True
    except Exception as e:
        print(f"✗ Erreur : {e}")
        return False


def download_bank_marketing():
    """Télécharge le dataset Bank Marketing depuis UCI"""
    print("\n=== Dataset Bank Marketing ===")
    try:
        # Essayer de télécharger depuis UCI
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
        zip_path = Path(__file__).parent / "bank.zip"
        
        print(f"Téléchargement de {url}...")
        urllib.request.urlretrieve(url, str(zip_path))
        print(f"✓ Fichier zip téléchargé")
        
        # Extraire
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(Path(__file__).parent)
        
        # Trouver le fichier CSV
        csv_files = list(Path(__file__).parent.glob("*.csv"))
        if csv_files:
            bank_file = csv_files[0]
            # Renommer si nécessaire
            target_file = Path(__file__).parent / "bank_marketing.csv"
            if bank_file != target_file:
                shutil.move(str(bank_file), str(target_file))
            
            df = pd.read_csv(target_file, sep=';')
            print(f"✓ Bank Marketing chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Nettoyer
            zip_path.unlink()
            return True
        else:
            print("✗ Fichier CSV non trouvé dans l'archive")
            return False
    except Exception as e:
        print(f"✗ Erreur : {e}")
        print("Note : Vous pouvez télécharger manuellement depuis https://archive.ics.uci.edu/dataset/222/bank+marketing")
        return False


def download_student_performance():
    """Télécharge le dataset Student Performance depuis UCI"""
    print("\n=== Dataset Student Performance ===")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        zip_path = Path(__file__).parent / "student.zip"
        
        print(f"Téléchargement de {url}...")
        urllib.request.urlretrieve(url, str(zip_path))
        print(f"✓ Fichier zip téléchargé")
        
        # Extraire
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(Path(__file__).parent)
        
        # Trouver les fichiers CSV
        csv_files = list(Path(__file__).parent.glob("*.csv"))
        if csv_files:
            for csv_file in csv_files:
                if 'student' in csv_file.name.lower() or 'mat' in csv_file.name.lower():
                    target_file = Path(__file__).parent / f"student_performance_{csv_file.name}"
                    if csv_file != target_file:
                        shutil.move(str(csv_file), str(target_file))
                    print(f"✓ {target_file.name} extrait")
            
            # Nettoyer
            zip_path.unlink()
            return True
        else:
            print("✗ Fichiers CSV non trouvés dans l'archive")
            return False
    except Exception as e:
        print(f"✗ Erreur : {e}")
        print("Note : Vous pouvez télécharger manuellement depuis https://archive.ics.uci.edu/dataset/320/student+performance")
        return False


def create_sample_datasets():
    """Crée des datasets d'exemple variés pour les tests"""
    print("\n=== Création de datasets d'exemple ===")
    
    # 1. Dataset de classification binaire équilibré
    np.random.seed(42)
    n_samples = 200
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    X2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    df_binary = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df_binary['target'] = y
    df_binary['target'] = df_binary['target'].map({0: 'class_A', 1: 'class_B'})
    
    file_path = Path(__file__).parent / "sample_binary_classification.csv"
    df_binary.to_csv(file_path, index=False)
    print(f"✓ Dataset classification binaire équilibré créé : {file_path.name}")
    
    # 2. Dataset de classification binaire DÉSÉQUILIBRÉ
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
    
    file_path = Path(__file__).parent / "sample_imbalanced_binary.csv"
    df_imbalanced.to_csv(file_path, index=False)
    print(f"✓ Dataset classification binaire DÉSÉQUILIBRÉ créé : {file_path.name}")
    
    # 3. Dataset de classification multiclasse (3 classes)
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
    
    file_path = Path(__file__).parent / "sample_multiclass_classification.csv"
    df_multiclass.to_csv(file_path, index=False)
    print(f"✓ Dataset classification multiclasse créé : {file_path.name}")
    
    # 4. Dataset de régression simple
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    
    df_regression = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    df_regression['target'] = y
    
    file_path = Path(__file__).parent / "sample_regression.csv"
    df_regression.to_csv(file_path, index=False)
    print(f"✓ Dataset régression créé : {file_path.name}")
    
    # 5. Dataset avec valeurs manquantes
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 4)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.3
    
    df_missing = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    df_missing['target'] = y
    
    # Ajouter des valeurs manquantes aléatoirement
    missing_indices = np.random.choice(df_missing.index, size=int(n_samples * 0.1), replace=False)
    df_missing.loc[missing_indices, 'feature_2'] = np.nan
    
    missing_indices = np.random.choice(df_missing.index, size=int(n_samples * 0.05), replace=False)
    df_missing.loc[missing_indices, 'feature_3'] = np.nan
    
    file_path = Path(__file__).parent / "sample_with_missing_values.csv"
    df_missing.to_csv(file_path, index=False)
    print(f"✓ Dataset avec valeurs manquantes créé : {file_path.name}")
    
    # 6. Dataset mixte (numérique + catégoriel)
    np.random.seed(42)
    n_samples = 250
    X_num = np.random.randn(n_samples, 2)
    categories = ['A', 'B', 'C']
    X_cat = np.random.choice(categories, size=n_samples)
    y = X_num[:, 0] + (X_cat == 'A').astype(int) * 2 + np.random.randn(n_samples) * 0.5
    
    df_mixed = pd.DataFrame(X_num, columns=['numeric_1', 'numeric_2'])
    df_mixed['categorical'] = X_cat
    df_mixed['target'] = y
    
    file_path = Path(__file__).parent / "sample_mixed_features.csv"
    df_mixed.to_csv(file_path, index=False)
    print(f"✓ Dataset avec features mixtes créé : {file_path.name}")
    
    # 7. Dataset de régression avec outliers
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    
    # Ajouter quelques outliers
    outlier_indices = np.random.choice(n_samples, size=10, replace=False)
    y[outlier_indices] += np.random.choice([-10, 10], size=10)
    
    df_outliers = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    df_outliers['target'] = y
    
    file_path = Path(__file__).parent / "sample_regression_outliers.csv"
    df_outliers.to_csv(file_path, index=False)
    print(f"✓ Dataset régression avec outliers créé : {file_path.name}")
    
    # 8. Dataset petit (pour tests rapides)
    np.random.seed(42)
    n_samples = 50
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] > 0).astype(int)
    
    df_small = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df_small['target'] = y
    df_small['target'] = df_small['target'].map({0: 'negative', 1: 'positive'})
    
    file_path = Path(__file__).parent / "sample_small.csv"
    df_small.to_csv(file_path, index=False)
    print(f"✓ Dataset petit (50 échantillons) créé : {file_path.name}")


def main():
    """Fonction principale"""
    print("=" * 60)
    print("Téléchargement des jeux de données de référence")
    print("=" * 60)
    
    # Créer le dossier si nécessaire
    datasets_dir = Path(__file__).parent
    datasets_dir.mkdir(exist_ok=True)
    
    results = {
        'Titanic': download_titanic(),
        'Iris': download_iris(),
        'California Housing': download_california_housing(),
        'Wine Quality': download_wine_quality(),
        'Bank Marketing': download_bank_marketing(),
        'Student Performance': download_student_performance()
    }
    
    # Créer des datasets d'exemple
    create_sample_datasets()
    
    # Résumé
    print("\n" + "=" * 60)
    print("Résumé du téléchargement")
    print("=" * 60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print("\n✓ Datasets disponibles dans le dossier 'datasets/'")
    print("\nVous pouvez maintenant utiliser ces datasets dans ChatAutoML-Bot !")


if __name__ == "__main__":
    main()

