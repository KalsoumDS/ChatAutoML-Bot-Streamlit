"""
Module de visualisations pour le chatbot
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
import streamlit as st

# Configuration des styles
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')


def create_progress_indicator(current_step: int, total_steps: int, step_name: str):
    """Crée un indicateur de progression"""
    progress = current_step / total_steps
    st.progress(progress, text=f"**Étape {current_step}/{total_steps}** : {step_name}")
    steps = []
    for i in range(1, total_steps + 1):
        if i < current_step:
            steps.append("✅")
        elif i == current_step:
            steps.append("🔄")
        else:
            steps.append("⏳")
    st.markdown(" ".join(steps))


def plot_target_distribution(y: pd.Series, task_type: str):
    """Crée un graphique de distribution de la cible"""
    fig = go.Figure()
    if task_type == 'classification':
        value_counts = y.value_counts()
        fig.add_trace(go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker_color='steelblue',
            text=value_counts.values,
            textposition='auto',
        ))
        fig.update_layout(
            title="Distribution de la variable cible (Classification)",
            xaxis_title="Classes",
            yaxis_title="Nombre d'échantillons",
            template="plotly_white",
            height=400
        )
    else:
        fig.add_trace(go.Histogram(
            x=y.values,
            nbinsx=30,
            marker_color='steelblue',
            opacity=0.7
        ))
        fig.update_layout(
            title="Distribution de la variable cible (Régression)",
            xaxis_title="Valeur",
            yaxis_title="Fréquence",
            template="plotly_white",
            height=400
        )
    return fig


def plot_missing_values(df_or_analysis):
    """
    Crée un graphique des valeurs manquantes.
    Accepte soit un DataFrame soit un dict d'analyse (rétrocompatibilité).
    """
    # Support des deux formes d'appel
    if isinstance(df_or_analysis, pd.DataFrame):
        df = df_or_analysis
        missing_data = []
        for col in df.columns:
            count = int(df[col].isnull().sum())
            if count > 0:
                missing_data.append({
                    'Colonne': col,
                    'Valeurs manquantes': count,
                    'Pourcentage': round(float(count / max(1, len(df)) * 100), 2)
                })
    else:
        # dict d'analyse legacy
        missing_data = []
        for col, info in df_or_analysis.get('missing_values', {}).items():
            if info['count'] > 0:
                missing_data.append({
                    'Colonne': col,
                    'Valeurs manquantes': info['count'],
                    'Pourcentage': info['percentage']
                })

    if not missing_data:
        return None

    df_missing = pd.DataFrame(missing_data)
    df_missing = df_missing.sort_values('Pourcentage', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_missing['Colonne'],
        y=df_missing['Pourcentage'],
        marker_color='coral',
        text=[f"{p:.1f}%" for p in df_missing['Pourcentage']],
        textposition='auto',
    ))
    fig.update_layout(
        title="Valeurs manquantes par colonne",
        xaxis_title="Colonnes",
        yaxis_title="Pourcentage de valeurs manquantes (%)",
        template="plotly_white",
        height=400,
        xaxis={'tickangle': -45}
    )
    return fig


def plot_feature_importance(feature_importance: Dict[str, float], top_n: int = 15):
    """Crée un graphique d'importance des features"""
    if not feature_importance:
        return None
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker_color='lightseagreen',
        text=[f"{imp:.4f}" for imp in importances],
        textposition='auto',
    ))
    fig.update_layout(
        title=f"Top {top_n} Features les plus importantes",
        xaxis_title="Importance",
        yaxis_title="Features",
        template="plotly_white",
        height=max(400, len(features) * 30)
    )
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = 'Best Score'):
    """Crée un graphique de comparaison des modèles"""
    fig = go.Figure()
    values = pd.to_numeric(comparison_df[metric], errors='coerce')
    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=values,
        marker_color='mediumpurple',
        text=[f"{val:.4f}" if not pd.isna(val) else "N/A" for val in values],
        textposition='auto',
    ))
    fig.update_layout(
        title="Comparaison des performances des modèles",
        xaxis_title="Modèles",
        yaxis_title=metric,
        template="plotly_white",
        height=400,
        xaxis={'tickangle': -45}
    )
    return fig


def plot_train_test_comparison(evaluation: Dict[str, Any], task_type: str):
    """Crée un graphique comparant train vs test"""
    train_metrics = evaluation.get('train_metrics', {})
    test_metrics = evaluation.get('test_metrics', {})
    if task_type == 'classification':
        metrics = ['accuracy', 'f1_macro', 'f1_weighted']
        metric_labels = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    else:
        metrics = ['r2', 'rmse', 'mae']
        metric_labels = ['R²', 'RMSE', 'MAE']
    available_metrics, available_labels, train_values, test_values = [], [], [], []
    for metric, label in zip(metrics, metric_labels):
        if metric in train_metrics and metric in test_metrics:
            available_metrics.append(metric)
            available_labels.append(label)
            train_values.append(train_metrics[metric])
            test_values.append(test_metrics[metric])
    if not available_metrics:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Train', x=available_labels, y=train_values,
        marker_color='lightblue',
        text=[f"{v:.4f}" for v in train_values], textposition='auto',
    ))
    fig.add_trace(go.Bar(
        name='Test', x=available_labels, y=test_values,
        marker_color='lightcoral',
        text=[f"{v:.4f}" for v in test_values], textposition='auto',
    ))
    fig.update_layout(
        title="Comparaison Train vs Test",
        xaxis_title="Métriques", yaxis_title="Score",
        template="plotly_white", height=400, barmode='group'
    )
    return fig


def plot_confusion_matrix(confusion_matrix_array: List[List[int]], class_names: Optional[List[str]] = None):
    """Crée une matrice de confusion"""
    if confusion_matrix_array is None:
        return None
    cm = np.array(confusion_matrix_array)
    if class_names is None:
        class_names = [f"Classe {i}" for i in range(len(cm))]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=class_names, y=class_names,
        colorscale='Blues', text=cm,
        texttemplate='%{text}', textfont={"size": 12},
        colorbar=dict(title="Nombre")
    ))
    fig.update_layout(
        title="Matrice de Confusion",
        xaxis_title="Prédictions", yaxis_title="Vraies valeurs",
        template="plotly_white", height=400
    )
    return fig


def create_summary_table(analysis: Dict[str, Any]) -> pd.DataFrame:
    """Crée un tableau récapitulatif du dataset"""
    summary_data = []
    for col, info in analysis.get('columns', {}).items():
        summary_data.append({
            'Colonne': str(col),
            'Type': str(info['type']),
            'Type Python': str(info['dtype']),
            'Valeurs uniques': int(info['unique_count']),
            'Valeurs manquantes': f"{info['missing_count']} ({info['missing_percentage']:.1f}%)"
        })
    return pd.DataFrame(summary_data)


def create_model_results_table(selection_result: Dict[str, Any], evaluation: Dict[str, Any]) -> pd.DataFrame:
    """Crée un tableau récapitulatif des résultats des modèles"""
    if 'comparison_summary' in selection_result:
        return selection_result['comparison_summary']
    return None


def create_metrics_table(evaluation: Dict[str, Any], task_type: str) -> pd.DataFrame:
    """Crée un tableau des métriques détaillées — toutes colonnes en str pour éviter types mixtes."""
    train_metrics = evaluation.get('train_metrics', {})
    test_metrics = evaluation.get('test_metrics', {})
    if task_type == 'classification':
        metrics = {
            'Accuracy': ('accuracy', 'accuracy'),
            'F1 Macro': ('f1_macro', 'f1_macro'),
            'F1 Weighted': ('f1_weighted', 'f1_weighted'),
            'ROC AUC': ('roc_auc', 'roc_auc')
        }
    else:
        metrics = {
            'R²': ('r2', 'r2'),
            'RMSE': ('rmse', 'rmse'),
            'MAE': ('mae', 'mae')
        }
    results = []
    for label, (train_key, test_key) in metrics.items():
        train_val = train_metrics.get(train_key, None)
        test_val = test_metrics.get(test_key, None)
        if train_val is not None or test_val is not None:
            train_str = f"{train_val:.4f}" if train_val is not None else "N/A"
            test_str = f"{test_val:.4f}" if test_val is not None else "N/A"
            ecart_str = f"{abs(train_val - test_val):.4f}" if train_val is not None and test_val is not None else "N/A"
            results.append({
                'Métrique': str(label),
                'Train': train_str,
                'Test': test_str,
                'Écart': ecart_str
            })
    return pd.DataFrame(results)


def create_dataset_head_table(df: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    """Crée un tableau avec les premières lignes du dataset — converti en str pour éviter types mixtes."""
    head = df.head(n_rows).copy()
    # Convertir toutes les colonnes object en str pour éviter le crash PyArrow
    for col in head.columns:
        if head[col].dtype == object:
            head[col] = head[col].astype(str)
    return head


def create_missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Crée un tableau récapitulatif des valeurs manquantes — types homogènes."""
    missing_data = []
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = round(float(missing_count / max(1, len(df)) * 100), 2)
        missing_data.append({
            'Colonne': str(col),
            'Valeurs manquantes': missing_count,       # int uniforme
            'Pourcentage (%)': missing_pct,            # float uniforme
            'Type': str(df[col].dtype)
        })
    return pd.DataFrame(missing_data)


def create_descriptive_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """Crée un tableau avec les statistiques descriptives — tous float, pas de str mélangés."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    stats = df[numeric_cols].describe().T.reset_index()
    stats.columns = ['Colonne', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    # Garder les valeurs numériques (float) — pas de conversion en str
    for col in ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors='coerce').round(4)
    stats['Colonne'] = stats['Colonne'].astype(str)
    return stats


def create_preprocessing_summary_table(X_before: pd.DataFrame, X_after: np.ndarray,
                                       preprocessor, column_types: Dict) -> pd.DataFrame:
    """Crée un tableau récapitulatif du preprocessing"""
    summary = []
    for col in column_types.get('numeric', []):
        summary.append({'Colonne': str(col), 'Type': 'Numérique', 'Traitement': 'Imputation (médiane) + Standardisation', 'Statut': '✓ Traité'})
    for col in column_types.get('categorical', []):
        summary.append({'Colonne': str(col), 'Type': 'Catégorielle', 'Traitement': 'Imputation (Unknown) + OneHotEncoding', 'Statut': '✓ Traité'})
    for col in column_types.get('boolean', []):
        summary.append({'Colonne': str(col), 'Type': 'Booléenne', 'Traitement': 'Imputation (mode) + Standardisation', 'Statut': '✓ Traité'})
    n_samples = X_after.shape[0] if len(X_after.shape) > 0 else 0
    n_features = X_after.shape[1] if len(X_after.shape) > 1 else 0
    summary.append({'Colonne': 'TOTAL', 'Type': f'{X_before.shape[1]} colonnes', 'Traitement': f'→ {n_features} features', 'Statut': f'✓ {n_samples} échantillons'})
    return pd.DataFrame(summary)


def create_train_test_split_table(X_train: np.ndarray, X_test: np.ndarray,
                                  y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Crée un tableau récapitulatif du split train/test"""
    total = X_train.shape[0] + X_test.shape[0]
    split_data = [
        {'Dataset': 'Train', 'Échantillons': X_train.shape[0], 'Features': X_train.shape[1], 'Proportion': f"{X_train.shape[0] / max(1, total) * 100:.1f}%"},
        {'Dataset': 'Test',  'Échantillons': X_test.shape[0],  'Features': X_test.shape[1],  'Proportion': f"{X_test.shape[0] / max(1, total) * 100:.1f}%"},
        {'Dataset': 'TOTAL', 'Échantillons': total,            'Features': X_train.shape[1], 'Proportion': '100%'},
    ]
    return pd.DataFrame(split_data)
