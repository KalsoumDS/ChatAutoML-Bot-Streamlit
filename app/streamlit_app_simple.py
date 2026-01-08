#!/usr/bin/env python3
"""
Interface Streamlit simple et fonctionnelle pour TabularAI
Version avec sidebar visible et menu déroulant
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
from typing import Any
from datetime import datetime
import io

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from automl.data_loader import DataLoader
from automl.preprocessing import Preprocessor
from automl.resampling import Resampler
from automl.model_zoo import ModelZoo
from automl.search import ModelSearcher
from automl.evaluation import ModelEvaluator
from automl.selection import ModelSelector
from llm.explainer import LLMExplainer
from app.visualizations import (
    create_progress_indicator, plot_target_distribution, plot_missing_values,
    plot_feature_importance, plot_model_comparison, plot_train_test_comparison,
    plot_confusion_matrix, create_summary_table, create_model_results_table,
    create_metrics_table, create_dataset_head_table, create_missing_values_table,
    create_descriptive_stats_table, create_preprocessing_summary_table,
    create_train_test_split_table
)
from app.chat_interface import detect_user_intent, generate_response

# Import de la configuration
try:
    import config.config as cfg
except ImportError:
    class DefaultConfig:
        MAX_FILE_SIZE_MB = 100
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        CV_FOLDS = 5
        SEARCH_METHOD = 'grid'
        N_ITER_RANDOM_SEARCH = 20
        RESAMPLING_STRATEGY = 'auto'
        LLM_PROVIDER = 'template'
    cfg = DefaultConfig()

# Récupérer le nom du chatbot depuis la config
try:
    CHATBOT_NAME = cfg.CHATBOT_NAME
    CHATBOT_DESCRIPTION = getattr(cfg, 'CHATBOT_DESCRIPTION', 'Votre assistant intelligent pour l\'AutoML sur données tabulaires')
    CHATBOT_ICON = getattr(cfg, 'CHATBOT_ICON', '🤖')
except AttributeError:
    CHATBOT_NAME = "TabularAI"
    CHATBOT_DESCRIPTION = "Votre assistant intelligent pour l'AutoML sur données tabulaires"
    CHATBOT_ICON = "🚀"

# Configuration de la page
st.set_page_config(
    page_title=CHATBOT_NAME,
    page_icon=CHATBOT_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS simple et fonctionnel
st.markdown("""
<style>
    .stApp {
        background: #f8fafc;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .stButton button {
        background: #3b82f6;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de la session
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.conversations = {
        "conv_default": {
            "name": "Discussion principale",
            "messages": []
        }
    }
    st.session_state.current_conversation_id = "conv_default"
    st.session_state.dataset = None
    st.session_state.target_column = None
    st.session_state.task_type = None
    st.session_state.analysis = None
    st.session_state.preprocessor = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.search_results = None
    st.session_state.selection_result = None
    st.session_state.evaluation = None
    st.session_state.metric = 'auto'
    st.session_state.llm_explainer = None
    st.session_state.current_file_name = None
    st.session_state.chatbot_name = CHATBOT_NAME
    st.session_state.chatbot_icon = CHATBOT_ICON

# Initialiser l'explainer LLM
if st.session_state.llm_explainer is None:
    if cfg.LLM_PROVIDER == 'ollama':
        st.session_state.llm_explainer = LLMExplainer(
            llm_provider=cfg.LLM_PROVIDER,
            model_name=getattr(cfg, 'OLLAMA_MODEL', 'llama2'),
            ollama_base_url=getattr(cfg, 'OLLAMA_BASE_URL', 'http://localhost:11434')
        )
    else:
        st.session_state.llm_explainer = LLMExplainer(
            llm_provider=cfg.LLM_PROVIDER,
            api_key=getattr(cfg, 'LLM_API_KEY', None)
        )


def _load_tabular_from_path(file_path: Path) -> pd.DataFrame:
    ext = file_path.suffix.lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    if ext in ['.tsv']:
        return pd.read_csv(file_path, sep='\t')
    if ext in ['.txt']:
        # Détection simple de séparateur (tab / ; / ,)
        with open(file_path, 'rb') as f:
            head = f.read(8192)
        try:
            sample = head.decode('utf-8')
        except Exception:
            sample = head.decode('latin-1', errors='ignore')
        seps = ['\t', ';', ',']
        counts = {s: sample.count(s) for s in seps}
        sep = max(counts, key=counts.get) if counts else ','
        return pd.read_csv(file_path, sep=sep)
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    if ext == '.parquet':
        return pd.read_parquet(file_path)
    if ext == '.json':
        return pd.read_json(file_path)
    raise ValueError(f"Format non supporté: {ext}")


def add_message(role: str, content: Any, content_type: str = "text", **kwargs):
    """Ajoute un message à l'historique"""
    message = {
        "role": role,
        "content": content,
        "content_type": content_type,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    st.session_state.messages.append(message)


def _render_bubble(role: str, html_content: str) -> str:
    justify = "flex-end" if role == "user" else "flex-start"
    bg = "#2563eb" if role == "user" else "#111827"
    border = "rgba(255,255,255,0.08)"
    text_color = "#ffffff" if role == "user" else "#e5e7eb"
    return (
        f"<div style='display:flex;justify-content:{justify};margin:0.25rem 0;'>"
        f"<div style='max-width:95%;padding:0.8rem 1rem;border-radius:1rem;"
        f"background:{bg};border:1px solid {border};box-shadow:0 1px 2px rgba(0,0,0,0.15);color:{text_color};"
        f"font-size:1rem;line-height:1.35;'>"
        f"{html_content}"
        "</div></div>"
    )


def _commentaire_valeurs_manquantes(df: pd.DataFrame, top_n: int = 3) -> str:
    if df is None or not hasattr(df, "isnull"):
        return ""

    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    total_missing = int(missing.sum())
    if missing.empty:
        return "Ce que j’ai trouvé : aucune valeur manquante détectée."
    top = missing.head(top_n)
    top_txt = ", ".join([f"`{col}` ({int(cnt)})" for col, cnt in top.items()])
    return (
        f"Ce que j’ai trouvé : **{total_missing:,}** valeurs manquantes au total. "
        f"Colonnes les plus touchées : {top_txt}."
    )


def _commentaire_confusion_matrix(cm) -> str:
    try:
        if cm is None:
            return ""
        arr = np.array(cm)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return ""
        total = float(arr.sum())
        if total <= 0:
            return ""
        correct = float(np.trace(arr))
        acc = correct / total
        return (
            f"Ce que j’ai trouvé : sur **{int(total):,}** prédictions, "
            f"**{int(correct):,}** sont correctes (exactitude ≈ **{acc * 100:.1f}%**). "
            "Les cases hors diagonale représentent les confusions entre classes."
        )
    except Exception:
        return ""


def _commentaire_classification_report(report: dict) -> str:
    try:
        if not report or not isinstance(report, dict):
            return ""
        acc = report.get('accuracy')
        macro = report.get('macro avg') or {}
        weighted = report.get('weighted avg') or {}
        macro_f1 = macro.get('f1-score')
        weighted_f1 = weighted.get('f1-score')
        parts = []
        if acc is not None:
            parts.append(f"exactitude ≈ **{float(acc) * 100:.1f}%**")
        if macro_f1 is not None:
            parts.append(f"F1 macro ≈ **{float(macro_f1):.3f}**")
        if weighted_f1 is not None:
            parts.append(f"F1 pondéré ≈ **{float(weighted_f1):.3f}**")
        if not parts:
            return "Ce que j’ai trouvé : le rapport résume précision/rappel/F1 par classe."
        return "Ce que j’ai trouvé : " + ", ".join(parts) + "."
    except Exception:
        return ""


def _commentaire_importance_variables(feature_importance, top_n: int = 3) -> str:
    try:
        if not feature_importance:
            return ""
        items = list(feature_importance.items())
        items = sorted(items, key=lambda x: abs(float(x[1])), reverse=True)
        top = items[:top_n]
        top_txt = ", ".join([f"`{name}`" for name, _ in top])
        return (
            f"Ce que j’ai trouvé : les variables les plus influentes semblent être {top_txt}. "
            "Cela indique quelles variables pèsent le plus dans la décision du modèle."
        )
    except Exception:
        return ""


def _commentaire_distribution_cible(y: pd.Series, task_type: str) -> str:
    try:
        if y is None:
            return ""
        if task_type == 'classification':
            vc = y.value_counts(dropna=False)
            n_classes = int(vc.shape[0])
            major = float(vc.max() / vc.sum()) if vc.sum() else 0.0
            return (
                f"Ce que j’ai trouvé : **{n_classes}** classe(s). "
                f"La classe majoritaire représente **{major * 100:.1f}%** des exemples."
            )
        return (
            "Ce que j’ai trouvé : la cible semble continue (régression). "
            "On observe sa répartition pour détecter des valeurs extrêmes ou une forte asymétrie."
        )
    except Exception:
        return ""


def _commentaire_split(X_train, X_test) -> str:
    try:
        n_train = int(X_train.shape[0])
        n_test = int(X_test.shape[0])
        total = n_train + n_test
        pct = (n_test / total * 100) if total else 0.0
        return (
            f"Ce que j’ai trouvé : séparation **entraînement/test** = "
            f"**{n_train:,} / {n_test:,}** lignes (test ≈ **{pct:.1f}%**)."
        )
    except Exception:
        return ""


def _commentaire_dimensions(X, X_transformed) -> str:
    try:
        before = int(X.shape[1])
        after = int(X_transformed.shape[1])
        if after > before:
            why = "(souvent dû à l’encodage des variables catégorielles)"
        elif after < before:
            why = "(réduction possible après traitement)"
        else:
            why = ""
        return f"Ce que j’ai trouvé : **{before}** variable(s) → **{after}** après transformation {why}."
    except Exception:
        return ""


def _commentaire_comparaison_modeles(comparison_df: pd.DataFrame) -> str:
    try:
        if comparison_df is None or comparison_df.empty:
            return ""
        score_col = 'Best Score' if 'Best Score' in comparison_df.columns else None
        if score_col is None:
            return "Ce que j’ai trouvé : comparaison des modèles disponible (voir tableau/graphique)."
        best_row = comparison_df.sort_values(score_col, ascending=False).iloc[0]
        model_name = None
        for cand in ['Model', 'Model Name', 'Modèle', 'Nom du modèle']:
            if cand in comparison_df.columns:
                model_name = str(best_row[cand])
                break
        best_score = float(best_row[score_col])
        if model_name:
            return f"Ce que j’ai trouvé : le meilleur modèle du comparatif est **{model_name}** (score ≈ **{best_score:.4f}**)."
        return f"Ce que j’ai trouvé : meilleur score dans le comparatif ≈ **{best_score:.4f}**."
    except Exception:
        return ""


def _commentaire_train_test(evaluation: dict, task_type: str) -> str:
    try:
        if evaluation is None:
            return ""
        train_metrics = evaluation.get('train_metrics') or {}
        test_metrics = evaluation.get('test_metrics') or {}
        metric_key = None
        for k in ['accuracy', 'f1', 'roc_auc', 'r2', 'mae', 'rmse']:
            if k in train_metrics and k in test_metrics:
                metric_key = k
                break
        if metric_key is None:
            return "Ce que j’ai trouvé : comparaison entraînement/test affichée ci-dessus."
        tr = float(train_metrics.get(metric_key))
        te = float(test_metrics.get(metric_key))
        gap = abs(tr - te)
        if task_type == 'regression':
            msg = "un écart important peut indiquer un sur-apprentissage (overfitting)."
        else:
            msg = "un écart important peut indiquer un sur-apprentissage (overfitting)."
        return (
            f"Ce que j’ai trouvé : `{metric_key}` entraînement ≈ **{tr:.4f}**, test ≈ **{te:.4f}** "
            f"(écart ≈ **{gap:.4f}**) — {msg}"
        )
    except Exception:
        return ""


def display_chat():
    """Affiche l'historique du chat"""
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        content_type = message.get("content_type", "text")

        # Avatar explicite pour un rendu plus "chatbot"
        chat_role = "assistant" if role == "assistant" else "user"
        avatar = st.session_state.get('chatbot_icon', CHATBOT_ICON) if role == "assistant" else "👤"

        with st.chat_message(chat_role, avatar=avatar):
            if content_type == "progress":
                current_step = message.get("step", 1)
                total_steps = message.get("total_steps", 1)
                step_name = message.get("step_name", "")
                create_progress_indicator(current_step, total_steps, step_name)
                if "text" in message:
                    st.markdown(message["text"], unsafe_allow_html=True)
            elif content_type == "dataframe":
                st.dataframe(message["content"], use_container_width=True, key=f"df_{idx}")
                if "text" in message:
                    st.markdown(_render_bubble(role, message["text"]), unsafe_allow_html=True)
            elif content_type == "plot":
                st.plotly_chart(message["content"], use_container_width=True, key=f"plot_{idx}")
                if "text" in message:
                    st.markdown(_render_bubble(role, message["text"]), unsafe_allow_html=True)
            elif content_type == "mixed":
                if "plots" in message:
                    for plot_idx, plot in enumerate(message["plots"]):
                        if plot is not None:
                            st.plotly_chart(plot, use_container_width=True, key=f"plot_{idx}_{plot_idx}")
                if "dataframes" in message:
                    for df_idx, df in enumerate(message["dataframes"]):
                        if df is not None:
                            st.dataframe(df, use_container_width=True, key=f"df_{idx}_{df_idx}")
                if "text" in message:
                    st.markdown(_render_bubble(role, message["text"]), unsafe_allow_html=True)
            else:
                st.markdown(_render_bubble(role, message["content"]), unsafe_allow_html=True)


def main():
    """Fonction principale"""

    def _main_simple():
        with st.sidebar:
            st.title("🎛️ Contrôles")

            action = st.selectbox(
                "Menu",
                ["Accueil", "Charger Dataset", "Analyser Données", "Lancer AutoML", "Résultats"],
                key="main_action"
            )

            with st.expander("🤖 Paramètres du chatbot", expanded=False):
                chatbot_name = st.text_input(
                    "Nom",
                    value=st.session_state.get('chatbot_name', CHATBOT_NAME),
                    key="chatbot_name_input"
                )
                if chatbot_name and chatbot_name != st.session_state.get('chatbot_name'):
                    st.session_state.chatbot_name = chatbot_name

                chatbot_icon = st.text_input(
                    "Icône (emoji)",
                    value=st.session_state.get('chatbot_icon', CHATBOT_ICON),
                    key="chatbot_icon_input"
                )
                if chatbot_icon and chatbot_icon != st.session_state.get('chatbot_icon'):
                    st.session_state.chatbot_icon = chatbot_icon

            st.divider()
            if st.session_state.get('dataset') is not None:
                st.success(f"✅ Dataset : {st.session_state.current_file_name}")
                st.caption(f"{st.session_state.dataset.shape[0]} lignes × {st.session_state.dataset.shape[1]} colonnes")
            else:
                st.warning("📁 Aucun dataset chargé")

            if st.session_state.get('dataset') is not None:
                if st.session_state.get('target_column'):
                    st.info(f"🎯 Cible : {st.session_state.target_column}")
                else:
                    opts = ["-- Sélectionner --"] + list(st.session_state.dataset.columns)
                    selected_target = st.selectbox("Colonne cible", opts, key="target_select")
                    if selected_target != "-- Sélectionner --" and selected_target != st.session_state.get('target_column'):
                        st.session_state.target_column = selected_target
                        loader = DataLoader(max_size_mb=cfg.MAX_FILE_SIZE_MB)
                        st.session_state.analysis = loader.analyze_dataset(st.session_state.dataset, selected_target)
                        st.session_state.task_type = loader.detect_task_type(st.session_state.dataset, selected_target)
                        st.rerun()

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🧹 Vider chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
            with c2:
                if st.button("🔄 Reset", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

            dataset = st.session_state.get('dataset')
            target_column = st.session_state.get('target_column')
            if dataset is not None and (not hasattr(dataset, "empty") or not dataset.empty) and target_column:
                if st.button("🚀 Lancer AutoML", use_container_width=True, type="primary"):
                    with st.spinner("⏳ Pipeline AutoML en cours..."):
                        run_automl()
                        st.rerun()

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;'>"
            f"<div style='font-size:2rem;line-height:2rem;'>{st.session_state.get('chatbot_icon', CHATBOT_ICON)}</div>"
            f"<div style='font-size:1.6rem;font-weight:800;'>{st.session_state.get('chatbot_name', CHATBOT_NAME)}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.caption(CHATBOT_DESCRIPTION)
        st.divider()

        if action == "Accueil":
            st.markdown("### 📌 Guide")
            st.markdown("1) Charger Dataset\n2) Choisir cible (sidebar)\n3) Lancer AutoML\n4) Résultats + questions")

        elif action == "Charger Dataset":
            st.title("📁 Charger un Dataset")
            source = st.radio(
                "Source des données",
                ["📂 Local (datasets)", "⬆️ Upload"]
            )
            datasets_dir = Path(__file__).parent.parent / "datasets"
            if source.startswith("📂"):
                supported = {'.csv', '.tsv', '.txt', '.xlsx', '.xls', '.parquet', '.json'}
                local_files = []
                if datasets_dir.exists():
                    local_files = [p for p in datasets_dir.rglob('*') if p.is_file() and p.suffix.lower() in supported]
                local_files = sorted(local_files, key=lambda p: str(p).lower())
                if local_files:
                    options = [str(p.relative_to(datasets_dir)) for p in local_files]
                    selected = st.selectbox("Fichier", options)
                    if st.button("✅ Charger", type="primary", use_container_width=True):
                        df = _load_tabular_from_path(datasets_dir / selected)
                        st.session_state.dataset = df
                        st.session_state.current_file_name = str(selected)
                        st.success(f"✅ Dataset chargé : {selected}")
                        st.dataframe(df.head(), use_container_width=True)
                else:
                    st.warning("⚠️ Aucun dataset trouvé dans datasets/")
            else:
                uploaded_file = st.file_uploader(
                    "Fichier",
                    type=['csv', 'tsv', 'txt', 'xlsx', 'xls', 'parquet', 'json']
                )
                if uploaded_file is not None:
                    raw = uploaded_file.getvalue()
                    ext = Path(uploaded_file.name).suffix.lower()
                    from io import BytesIO, StringIO
                    if ext == '.csv':
                        df = pd.read_csv(BytesIO(raw))
                    elif ext == '.tsv':
                        df = pd.read_csv(BytesIO(raw), sep='\t')
                    elif ext == '.txt':
                        try:
                            sample = raw[:8192].decode('utf-8')
                        except Exception:
                            sample = raw[:8192].decode('latin-1', errors='ignore')
                        seps = ['\t', ';', ',']
                        counts = {s: sample.count(s) for s in seps}
                        sep = max(counts, key=counts.get) if counts else ','
                        df = pd.read_csv(BytesIO(raw), sep=sep)
                    elif ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(BytesIO(raw))
                    elif ext == '.parquet':
                        df = pd.read_parquet(BytesIO(raw))
                    elif ext == '.json':
                        df = pd.read_json(StringIO(raw.decode('utf-8')))
                    else:
                        st.error(f"❌ Format non supporté: {ext}")
                        df = None
                    if df is not None:
                        st.session_state.dataset = df
                        st.session_state.current_file_name = uploaded_file.name
                        st.success(f"✅ Dataset chargé : {uploaded_file.name}")
                        st.dataframe(df.head(), use_container_width=True)

        elif action == "Analyser Données":
            st.title("🔍 Analyse des Données")
            if st.session_state.get('dataset') is None:
                st.warning("⚠️ Chargez d'abord un dataset")
            else:
                df = st.session_state.dataset
                st.dataframe(df.head(), use_container_width=True)

        elif action == "Lancer AutoML":
            st.title("🚀 Pipeline AutoML")
            st.info("Choisissez la cible dans la sidebar puis lancez AutoML.")

        elif action == "Résultats":
            st.title("📈 Résultats AutoML")
            if st.session_state.get('evaluation') is None:
                st.warning("⚠️ Lancez d'abord AutoML")

        st.divider()
        st.subheader("💬 Chat")
        if not st.session_state.messages:
            add_message("assistant", f"Bonjour, je suis **{st.session_state.get('chatbot_name', CHATBOT_NAME)}**.")
        display_chat()
        if prompt := st.chat_input("Écrivez ici..."):
            add_message("user", prompt)
            intent = detect_user_intent(prompt)
            context = {
                'analysis': st.session_state.get('analysis'),
                'task_type': st.session_state.get('task_type'),
                'selection_result': st.session_state.get('selection_result'),
                'evaluation': st.session_state.get('evaluation'),
                'dataset': st.session_state.get('dataset'),
                'target_column': st.session_state.get('target_column')
            }
            response = generate_response(intent, context, st.session_state.llm_explainer, prompt)
            add_message("assistant", response, content_type="text")
            st.rerun()

    def _main_good():
        # Sidebar: navigation + données
        with st.sidebar:
            st.markdown("### 🧭 Navigation")

            action = st.selectbox(
                "Menu",
                ["Accueil", "Charger Dataset", "Analyser Données", "Lancer AutoML", "Résultats"],
                key="main_action"
            )

            with st.expander("🤖 Chatbot", expanded=False):
                chatbot_name = st.text_input(
                    "Nom",
                    value=st.session_state.get('chatbot_name', CHATBOT_NAME),
                    key="chatbot_name_input"
                )
                if chatbot_name and chatbot_name != st.session_state.get('chatbot_name'):
                    st.session_state.chatbot_name = chatbot_name

                chatbot_icon = st.text_input(
                    "Icône (emoji)",
                    value=st.session_state.get('chatbot_icon', CHATBOT_ICON),
                    key="chatbot_icon_input"
                )
                if chatbot_icon and chatbot_icon != st.session_state.get('chatbot_icon'):
                    st.session_state.chatbot_icon = chatbot_icon

            st.divider()

            with st.expander("📁 Dataset", expanded=True):
                if st.session_state.get('dataset') is not None:
                    st.success(f"Chargé : {st.session_state.current_file_name}")
                    st.caption(f"{st.session_state.dataset.shape[0]} lignes × {st.session_state.dataset.shape[1]} colonnes")
                    if st.button("🔁 Changer de dataset", use_container_width=True):
                        st.session_state.main_action = "Charger Dataset"
                        st.rerun()
                else:
                    st.warning("Aucun dataset")
                    if st.button("➕ Charger maintenant", type="primary", use_container_width=True):
                        st.session_state.main_action = "Charger Dataset"
                        st.rerun()

            with st.expander("🎯 Cible", expanded=True):
                if st.session_state.get('dataset') is None:
                    st.info("Chargez un dataset d'abord")
                else:
                    if st.session_state.get('target_column'):
                        st.success(f"Cible : {st.session_state.target_column}")
                        if st.button("✏️ Modifier la cible", use_container_width=True):
                            st.session_state.target_column = None
                            st.rerun()
                    else:
                        opts = ["-- Sélectionner --"] + list(st.session_state.dataset.columns)
                        selected_target = st.selectbox("Colonne cible", opts, key="target_select")
                        if selected_target != "-- Sélectionner --" and selected_target != st.session_state.get('target_column'):
                            st.session_state.target_column = selected_target
                            loader = DataLoader(max_size_mb=cfg.MAX_FILE_SIZE_MB)
                            st.session_state.analysis = loader.analyze_dataset(st.session_state.dataset, selected_target)
                            st.session_state.task_type = loader.detect_task_type(st.session_state.dataset, selected_target)
                            st.rerun()

                    if st.session_state.get('task_type'):
                        st.caption(f"Type détecté : **{st.session_state.task_type}**")

            with st.expander("🚀 AutoML", expanded=True):
                dataset = st.session_state.get('dataset')
                target_column = st.session_state.get('target_column')
                ready = dataset is not None and (not hasattr(dataset, "empty") or not dataset.empty) and target_column

                if not ready:
                    st.info("Dataset + cible requis")
                else:
                    if st.button("🚀 Lancer AutoML", use_container_width=True, type="primary"):
                        with st.spinner("⏳ Pipeline AutoML en cours..."):
                            try:
                                run_automl()
                                st.session_state.main_action = "Résultats"
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Erreur : {str(e)}")

            with st.expander("🧹 Actions", expanded=False):
                if st.button("🧹 Vider chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

                if st.button("🔄 Reset complet", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

        # Layout principal: contenu + chat à droite (persistant)
        col_main, col_chat = st.columns([2.2, 1.8], gap="large")

        with col_main:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;'>"
                f"<div style='font-size:2rem;line-height:2rem;'>{st.session_state.get('chatbot_icon', CHATBOT_ICON)}</div>"
                f"<div style='font-size:1.6rem;font-weight:800;'>{st.session_state.get('chatbot_name', CHATBOT_NAME)}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.caption(CHATBOT_DESCRIPTION)
            st.divider()

            if action == "Accueil":
                st.markdown("### 📌 Guide du projet")
                st.markdown(
                    "1) **Charger Dataset**\n"
                    "2) **Choisir la cible** (dans la sidebar)\n"
                    "3) **Lancer AutoML**\n"
                    "4) Lire les **Résultats** et poser vos questions dans le chat"
                )

            elif action == "Charger Dataset":
                st.title("📁 Charger un Dataset")
                source = st.radio(
                    "Source des données",
                    ["📂 Choisir un dataset local (dossier datasets)", "⬆️ Importer un fichier (upload)"]
                )

                datasets_dir = Path(__file__).parent.parent / "datasets"
                if source.startswith("📂"):
                    supported = {'.csv', '.tsv', '.txt', '.xlsx', '.xls', '.parquet', '.json'}
                    local_files = []
                    if datasets_dir.exists():
                        local_files = [p for p in datasets_dir.rglob('*') if p.is_file() and p.suffix.lower() in supported]
                    local_files = sorted(local_files, key=lambda p: str(p).lower())
                    if not local_files:
                        st.warning("⚠️ Aucun dataset tabulaire trouvé dans datasets/")
                    else:
                        options = [str(p.relative_to(datasets_dir)) for p in local_files]
                        selected = st.selectbox("Choisir un fichier", options)
                        if st.button("✅ Charger", type="primary", use_container_width=True):
                            try:
                                df = _load_tabular_from_path(datasets_dir / selected)
                                st.session_state.dataset = df
                                st.session_state.current_file_name = str(selected)
                                st.success(f"✅ Dataset chargé : {selected}")
                                st.dataframe(df.head(), use_container_width=True)
                            except Exception as e:
                                st.error(f"❌ Erreur lors du chargement : {str(e)}")
                else:
                    uploaded_file = st.file_uploader(
                        "Sélectionnez votre fichier",
                        type=['csv', 'tsv', 'txt', 'xlsx', 'xls', 'parquet', 'json'],
                        help="Formats supportés : CSV, TSV, TXT, Excel, Parquet, JSON"
                    )
                    if uploaded_file is not None:
                        try:
                            raw = uploaded_file.getvalue()
                            max_bytes = int(cfg.MAX_FILE_SIZE_MB) * 1024 * 1024
                            if len(raw) > max_bytes:
                                raise ValueError(
                                    f"Fichier trop volumineux ({len(raw) / (1024 * 1024):.2f} Mo). Maximum: {cfg.MAX_FILE_SIZE_MB} Mo"
                                )

                            ext = Path(uploaded_file.name).suffix.lower()
                            from io import BytesIO, StringIO
                            if ext == '.csv':
                                df = pd.read_csv(BytesIO(raw))
                            elif ext == '.tsv':
                                df = pd.read_csv(BytesIO(raw), sep='\t')
                            elif ext == '.txt':
                                try:
                                    sample = raw[:8192].decode('utf-8')
                                except Exception:
                                    sample = raw[:8192].decode('latin-1', errors='ignore')
                                seps = ['\t', ';', ',']
                                counts = {s: sample.count(s) for s in seps}
                                sep = max(counts, key=counts.get) if counts else ','
                                df = pd.read_csv(BytesIO(raw), sep=sep)
                            elif ext in ['.xlsx', '.xls']:
                                df = pd.read_excel(BytesIO(raw))
                            elif ext == '.parquet':
                                df = pd.read_parquet(BytesIO(raw))
                            elif ext == '.json':
                                df = pd.read_json(StringIO(raw.decode('utf-8')))
                            else:
                                raise ValueError(f"Format non supporté: {ext}")

                            st.session_state.dataset = df
                            st.session_state.current_file_name = uploaded_file.name
                            st.success(f"✅ Dataset chargé : {uploaded_file.name}")
                            st.dataframe(df.head(), use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ Erreur lors du chargement : {str(e)}")

            elif action == "Analyser Données":
                st.title("🔍 Analyse des Données")
                if st.session_state.get('dataset') is None:
                    st.warning("⚠️ Chargez d'abord un dataset")
                else:
                    df = st.session_state.dataset
                    st.subheader("📊 Aperçu")
                    st.dataframe(df.head(20), use_container_width=True)

            elif action == "Lancer AutoML":
                st.title("🚀 Pipeline AutoML")
                if st.session_state.get('dataset') is None:
                    st.warning("⚠️ Chargez d'abord un dataset")
                elif st.session_state.get('target_column') is None:
                    st.warning("⚠️ Sélectionnez une colonne cible (sidebar)")
                else:
                    st.info("Cliquez sur **🚀 Lancer AutoML** dans la sidebar.")

            elif action == "Résultats":
                st.title("📈 Résultats AutoML")
                if st.session_state.get('evaluation') is None:
                    st.warning("⚠️ Lancez d'abord AutoML")
                else:
                    evaluation = st.session_state.evaluation
                    selection_result = st.session_state.selection_result
                    if selection_result is not None:
                        st.success(
                            f"🏆 **{selection_result.get('best_model_name', 'N/A')}** — score : {selection_result.get('best_score', 0):.4f}"
                        )

        with col_chat:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
                f"<div style='font-size:1.3rem;'>{st.session_state.get('chatbot_icon', CHATBOT_ICON)}</div>"
                f"<div style='font-size:1.05rem;font-weight:800;'>Chat</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.divider()

            if not st.session_state.messages:
                add_message(
                    "assistant",
                    (
                        f"Bonjour, je suis **{st.session_state.get('chatbot_name', CHATBOT_NAME)}**.\n\n"
                        "Je peux vous aider à :\n"
                        "- comprendre votre dataset\n"
                        "- choisir une cible\n"
                        "- interpréter les résultats AutoML"
                    ),
                    content_type="text"
                )

            with st.container(height=560, border=False):
                display_chat()
            if prompt := st.chat_input("Écrivez ici..."):
                add_message("user", prompt)

                intent = detect_user_intent(prompt)
                context = {
                    'analysis': st.session_state.get('analysis'),
                    'task_type': st.session_state.get('task_type'),
                    'selection_result': st.session_state.get('selection_result'),
                    'evaluation': st.session_state.get('evaluation'),
                    'dataset': st.session_state.get('dataset'),
                    'target_column': st.session_state.get('target_column')
                }

                response = generate_response(intent, context, st.session_state.llm_explainer, prompt)
                add_message("assistant", response, content_type="text")
                st.rerun()

    return _main_good()

    # Sidebar minimale
    with st.sidebar:
        st.title("⚙️ Données")
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        with st.expander("🤖 Paramètres", expanded=False):
            chatbot_name = st.text_input(
                "Nom",
                value=st.session_state.get('chatbot_name', CHATBOT_NAME),
                key="chatbot_name_input"
            )
            if chatbot_name and chatbot_name != st.session_state.get('chatbot_name'):
                st.session_state.chatbot_name = chatbot_name

            chatbot_icon = st.text_input(
                "Icône (emoji)",
                value=st.session_state.get('chatbot_icon', CHATBOT_ICON),
                key="chatbot_icon_input"
            )
            if chatbot_icon and chatbot_icon != st.session_state.get('chatbot_icon'):
                st.session_state.chatbot_icon = chatbot_icon

        st.divider()
        if st.session_state.get('dataset') is not None:
            st.success(f"✅ Dataset : {st.session_state.current_file_name}")
            st.caption(f"{st.session_state.dataset.shape[0]} lignes × {st.session_state.dataset.shape[1]} colonnes")
            if st.session_state.get('target_column'):
                st.info(f"🎯 Cible : {st.session_state.target_column}")
        else:
            st.warning("📁 Aucun dataset")

    # Flux guidé (piloté par le chatbot)
    if 'flow_step' not in st.session_state:
        st.session_state.flow_step = None

    dataset = st.session_state.get('dataset')
    target_column = st.session_state.get('target_column')
    evaluation = st.session_state.get('evaluation')

    if dataset is None:
        recommended_step = "charger"
    elif target_column is None:
        recommended_step = "cible"
    elif evaluation is None:
        recommended_step = "automl"
    else:
        recommended_step = "resultats"

    if st.session_state.flow_step is None:
        st.session_state.flow_step = recommended_step

    # Layout style messagerie: conversations (gauche) + chat (centre) + info/actions (droite)
    col_convos, col_chat, col_info = st.columns([1.2, 2.8, 1.2], gap="large")

    with col_convos:
        st.markdown("### Chats")
        search = st.text_input("Rechercher", value="", key="conv_search")

        if st.button("➕ Nouvelle discussion", use_container_width=True):
            new_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            conversations = st.session_state.get('conversations', {})
            conversations[new_id] = {"name": f"Discussion {len(conversations)+1}", "messages": []}
            st.session_state.conversations = conversations
            st.session_state.current_conversation_id = new_id
            st.rerun()

        conversations = st.session_state.get('conversations', {})
        items = []
        for cid, meta in conversations.items():
            name = meta.get('name', cid)
            if search and search.lower() not in name.lower():
                continue
            items.append((cid, name))

        for cid, name in items:
            is_active = cid == st.session_state.get('current_conversation_id')
            label = f"➡️ {name}" if is_active else name
            if st.button(label, use_container_width=True, key=f"conv_btn_{cid}"):
                st.session_state.current_conversation_id = cid
                st.session_state.messages = conversations[cid].get('messages', [])
                st.rerun()

    with col_info:
        st.markdown("### Info")
        st.caption("Données & actions")
        st.divider()

        if st.button("🧹 Vider l'historique", use_container_width=True):
            conv_id = st.session_state.get('current_conversation_id', 'conv_default')
            conversations = st.session_state.get('conversations', {})
            if conv_id in conversations:
                conversations[conv_id]['messages'] = []
                st.session_state.conversations = conversations
            st.session_state.messages = []
            st.rerun()

        step = st.session_state.flow_step
        st.markdown(f"**Étape** : `{step}`")
        st.divider()

        b1, b2 = st.columns(2)
        with b1:
            if st.button("📁 Charger", use_container_width=True):
                st.session_state.flow_step = "charger"
                st.rerun()
        with b2:
            if st.button("🎯 Cible", use_container_width=True):
                st.session_state.flow_step = "cible"
                st.rerun()
        b3, b4 = st.columns(2)
        with b3:
            if st.button("🚀 AutoML", use_container_width=True):
                st.session_state.flow_step = "automl"
                st.rerun()
        with b4:
            if st.button("📈 Résultats", use_container_width=True):
                st.session_state.flow_step = "resultats"
                st.rerun()

        st.divider()
        if st.session_state.get('dataset') is not None:
            st.success("✅ Dataset chargé")
            st.caption(st.session_state.get('current_file_name', ''))
        if st.session_state.get('target_column'):
            st.info(f"🎯 Cible : {st.session_state.get('target_column')}")

        if step == "automl" and st.session_state.get('dataset') is not None and st.session_state.get('target_column') is not None:
            if st.button("▶️ Lancer AutoML maintenant", type="primary", use_container_width=True):
                with st.spinner("⏳ Pipeline AutoML en cours..."):
                    try:
                        run_automl()
                        st.session_state.flow_step = "resultats"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur : {str(e)}")

        st.divider()

        # Contrôles d'étape (dans le panneau Info)
        if step == "charger":
            st.subheader("📁 Charger")
            source = st.radio(
                "Source",
                ["📂 Local (dossier datasets)", "⬆️ Upload"],
                key="source_info"
            )

            datasets_dir = Path(__file__).parent.parent / "datasets"
            if source.startswith("📂"):
                supported = {'.csv', '.tsv', '.txt', '.xlsx', '.xls', '.parquet', '.json'}
                local_files = []
                if datasets_dir.exists():
                    local_files = [p for p in datasets_dir.rglob('*') if p.is_file() and p.suffix.lower() in supported]
                local_files = sorted(local_files, key=lambda p: str(p).lower())

                if not local_files:
                    st.warning("⚠️ Aucun dataset trouvé")
                else:
                    options = [str(p.relative_to(datasets_dir)) for p in local_files]
                    selected = st.selectbox("Fichier", options, key="local_file_info")
                    if st.button("✅ Charger", type="primary", use_container_width=True, key="load_local_info"):
                        try:
                            df = _load_tabular_from_path(datasets_dir / selected)
                            st.session_state.dataset = df
                            st.session_state.current_file_name = str(selected)
                            st.session_state.flow_step = "cible"
                            add_message(
                                "assistant",
                                f"Dataset chargé : `{selected}`. Choisissez maintenant la colonne cible.",
                                content_type="text"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {str(e)}")
            else:
                uploaded_file = st.file_uploader(
                    "Fichier",
                    type=['csv', 'tsv', 'txt', 'xlsx', 'xls', 'parquet', 'json'],
                    key="uploader_info"
                )
                if uploaded_file is not None:
                    try:
                        raw = uploaded_file.getvalue()
                        max_bytes = int(cfg.MAX_FILE_SIZE_MB) * 1024 * 1024
                        if len(raw) > max_bytes:
                            raise ValueError(
                                f"Fichier trop volumineux ({len(raw) / (1024 * 1024):.2f} Mo). Maximum: {cfg.MAX_FILE_SIZE_MB} Mo"
                            )
                        ext = Path(uploaded_file.name).suffix.lower()
                        from io import BytesIO, StringIO
                        if ext == '.csv':
                            df = pd.read_csv(BytesIO(raw))
                        elif ext == '.tsv':
                            df = pd.read_csv(BytesIO(raw), sep='\t')
                        elif ext == '.txt':
                            try:
                                sample = raw[:8192].decode('utf-8')
                            except Exception:
                                sample = raw[:8192].decode('latin-1', errors='ignore')
                            seps = ['\t', ';', ',']
                            counts = {s: sample.count(s) for s in seps}
                            sep = max(counts, key=counts.get) if counts else ','
                            df = pd.read_csv(BytesIO(raw), sep=sep)
                        elif ext in ['.xlsx', '.xls']:
                            df = pd.read_excel(BytesIO(raw))
                        elif ext == '.parquet':
                            df = pd.read_parquet(BytesIO(raw))
                        elif ext == '.json':
                            df = pd.read_json(StringIO(raw.decode('utf-8')))
                        else:
                            raise ValueError(f"Format non supporté: {ext}")

                        st.session_state.dataset = df
                        st.session_state.current_file_name = uploaded_file.name
                        st.session_state.flow_step = "cible"
                        add_message(
                            "assistant",
                            f"Dataset chargé : `{uploaded_file.name}`. Choisissez maintenant la colonne cible.",
                            content_type="text"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

        elif step == "cible":
            st.subheader("🎯 Cible")
            if st.session_state.get('dataset') is None:
                st.warning("⚠️ Chargez un dataset")
            else:
                df = st.session_state.dataset
                current_target = st.session_state.get('target_column')
                options = ["-- Sélectionner --"] + list(df.columns)
                default_idx = options.index(current_target) if current_target in options else 0
                selected_target = st.selectbox(
                    "Colonne cible",
                    options,
                    index=default_idx,
                    key="target_select_info"
                )
                if selected_target != "-- Sélectionner --" and selected_target != current_target:
                    st.session_state.target_column = selected_target
                    loader = DataLoader(max_size_mb=cfg.MAX_FILE_SIZE_MB)
                    analysis = loader.analyze_dataset(df, selected_target)
                    st.session_state.analysis = analysis
                    task_type = loader.detect_task_type(df, selected_target)
                    st.session_state.task_type = task_type
                    add_message(
                        "assistant",
                        f"Cible définie : `{selected_target}`. Type de tâche détecté : **{task_type}**. Vous pouvez lancer AutoML.",
                        content_type="text"
                    )
                    st.rerun()

        elif step == "resultats":
            st.subheader("📈 Résultats")
            if st.session_state.get('evaluation') is None or st.session_state.get('selection_result') is None:
                st.warning("⚠️ Aucun résultat pour l’instant. Lancez AutoML.")
            else:
                evaluation = st.session_state.evaluation
                selection_result = st.session_state.selection_result
                df = st.session_state.get('dataset')
                target_col = st.session_state.get('target_column')
                task_type = st.session_state.get('task_type')
                analysis = st.session_state.get('analysis')

                st.markdown("**🏆 Meilleur modèle**")
                st.success(f"{selection_result.get('best_model_name', 'N/A')} — score : {selection_result.get('best_score', 0):.4f}")

                comparison_df = selection_result.get('comparison_summary') if isinstance(selection_result, dict) else None
                if comparison_df is not None:
                    st.markdown("**📌 Comparaison des modèles**")
                    st.dataframe(comparison_df, use_container_width=True)
                    insight = _commentaire_comparaison_modeles(comparison_df)
                    if insight:
                        st.caption(insight)
                    try:
                        score_col = 'Best Score' if 'Best Score' in comparison_df.columns else comparison_df.columns[1]
                        st.plotly_chart(plot_model_comparison(comparison_df, score_col), use_container_width=True)
                    except Exception:
                        pass

                metrics_df = create_metrics_table(evaluation, task_type)
                if metrics_df is not None:
                    st.markdown("**📊 Métriques**")
                    st.dataframe(metrics_df, use_container_width=True)
                    metrics_insight = _commentaire_train_test(evaluation, task_type)
                    if metrics_insight:
                        st.caption(metrics_insight)

                train_test_plot = plot_train_test_comparison(evaluation, task_type)
                if train_test_plot:
                    st.markdown("**📉 Entraînement vs Test**")
                    st.plotly_chart(train_test_plot, use_container_width=True)

                if task_type == 'classification':
                    if evaluation.get('confusion_matrix'):
                        st.markdown("**🔍 Matrice de confusion**")
                        cm_plot = plot_confusion_matrix(evaluation['confusion_matrix'])
                        if cm_plot:
                            st.plotly_chart(cm_plot, use_container_width=True)
                        cm_insight = _commentaire_confusion_matrix(evaluation.get('confusion_matrix'))
                        if cm_insight:
                            st.caption(cm_insight)

                    if evaluation.get('classification_report'):
                        st.markdown("**🧾 Rapport de classification**")
                        try:
                            report_df = pd.DataFrame(evaluation['classification_report']).T
                            st.dataframe(report_df, use_container_width=True)
                            rep_insight = _commentaire_classification_report(evaluation['classification_report'])
                            if rep_insight:
                                st.caption(rep_insight)
                        except Exception:
                            pass

                try:
                    preprocessor = st.session_state.get('preprocessor')
                    if preprocessor is not None and selection_result and selection_result.get('best_model') is not None:
                        evaluator = ModelEvaluator(task_type=task_type)
                        feature_names = getattr(preprocessor, 'feature_names', None)
                        if feature_names is not None:
                            feature_importance = evaluator.get_feature_importance(selection_result['best_model'], feature_names)
                            if feature_importance:
                                st.markdown("**🧠 Importance des variables**")
                                importance_plot = plot_feature_importance(feature_importance)
                                if importance_plot:
                                    st.plotly_chart(importance_plot, use_container_width=True)
                                imp_insight = _commentaire_importance_variables(feature_importance)
                                if imp_insight:
                                    st.caption(imp_insight)
                except Exception:
                    pass

                if df is not None and target_col is not None and target_col in df.columns and task_type is not None:
                    st.divider()
                    st.markdown("**🗂️ Rappel dataset**")
                    st.caption(f"{df.shape[0]:,} lignes × {df.shape[1]} colonnes — cible : `{target_col}`")
                    try:
                        st.plotly_chart(plot_target_distribution(df[target_col], task_type), use_container_width=True)
                        tgt_insight = _commentaire_distribution_cible(df[target_col], task_type)
                        if tgt_insight:
                            st.caption(tgt_insight)
                    except Exception:
                        pass
                    try:
                        if analysis is not None:
                            missing_plot = plot_missing_values(analysis)
                            if missing_plot:
                                st.plotly_chart(missing_plot, use_container_width=True)
                                miss_insight = _commentaire_valeurs_manquantes(df)
                                if miss_insight:
                                    st.caption(miss_insight)
                    except Exception:
                        pass


    with col_chat:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;'>"
            f"<div style='font-size:2rem;line-height:2rem;'>{st.session_state.get('chatbot_icon', CHATBOT_ICON)}</div>"
            f"<div style='font-size:1.6rem;font-weight:800;'>{st.session_state.get('chatbot_name', CHATBOT_NAME)}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.caption(CHATBOT_DESCRIPTION)
        st.divider()

        # Message d'accueil si aucune conversation encore
        if not _get_active_messages():
            add_message(
                "assistant",
                (
                    f"Bonjour, je suis **{st.session_state.get('chatbot_name', CHATBOT_NAME)}**.\n\n"
                    "Je vais vous guider pas à pas :\n"
                    "1) Charger un dataset\n"
                    "2) Choisir la colonne cible\n"
                    "3) Lancer AutoML\n"
                    "4) Lire les résultats\n\n"
                    "Cliquez sur **📁 Charger** dans le panneau Info pour commencer."
                ),
                content_type="text"
            )

        display_chat()

        if prompt := st.chat_input("Écrivez ici..."):
            add_message("user", prompt)

            intent = detect_user_intent(prompt)
            context = {
                'analysis': st.session_state.get('analysis'),
                'task_type': st.session_state.get('task_type'),
                'selection_result': st.session_state.get('selection_result'),
                'evaluation': st.session_state.get('evaluation'),
                'dataset': st.session_state.get('dataset'),
                'target_column': st.session_state.get('target_column')
            }

            response = generate_response(intent, context, st.session_state.llm_explainer, prompt)
            add_message("assistant", response, content_type="text")
            st.rerun()


def run_automl():
    """Exécute le pipeline AutoML complet"""
    df = st.session_state.get('dataset')
    target_col = st.session_state.get('target_column')
    task_type = st.session_state.get('task_type')
    metric = st.session_state.get('metric', 'auto')

    if df is None or target_col is None:
        raise ValueError("Dataset et colonne cible doivent être définis")

    total_steps = 6

    # Étape 0 (contexte) : rappeler les infos dataset
    try:
        analysis = st.session_state.get('analysis')
        if analysis is None:
            loader = DataLoader(max_size_mb=cfg.MAX_FILE_SIZE_MB)
            analysis = loader.analyze_dataset(df, target_col)
            st.session_state.analysis = analysis
            if task_type is None:
                task_type = loader.detect_task_type(df, target_col)
                st.session_state.task_type = task_type

        intro_text = (
            "# 🧾 Pré-analyse du dataset\n\n"
            f"**Fichier** : {st.session_state.get('current_file_name', 'N/A')}\n\n"
            f"**Dimensions** : {df.shape[0]:,} lignes × {df.shape[1]} colonnes\n\n"
            f"**Colonne cible** : `{target_col}`\n\n"
            f"**Type de tâche détecté** : **{task_type}**\n\n"
            "Ci-dessous : aperçu, valeurs manquantes et distribution de la cible.\n\n"
            f"{_commentaire_distribution_cible(df[target_col], task_type)}\n\n"
            f"{_commentaire_valeurs_manquantes(df)}"
        )

        plots = []
        try:
            plots.append(plot_target_distribution(df[target_col], task_type))
        except Exception:
            pass
        try:
            missing_plot = plot_missing_values(analysis)
            if missing_plot:
                plots.append(missing_plot)
        except Exception:
            pass

        dataframes = []
        try:
            dataframes.append(create_dataset_head_table(df, n_rows=10))
        except Exception:
            pass
        try:
            dataframes.append(create_missing_values_table(df))
        except Exception:
            pass

        add_message(
            "assistant",
            intro_text,
            content_type="mixed",
            text=intro_text,
            plots=plots,
            dataframes=dataframes
        )
    except Exception:
        pass

    # ============================================
    # ÉTAPE 1 : PREPROCESSING
    # ============================================
    add_message("assistant",
        "<div style=\"font-size:1.35rem;font-weight:800;margin:0.25rem 0 0.5rem 0;\">🔧 Étape 1/6 — Prétraitement des données</div>"
        "<div style=\"margin:0 0 0.5rem 0;\"><b>Objectif</b> : préparer les données pour l'entraînement</div>"
        "<div><b>Traitement en cours...</b></div>",
        content_type="progress",
        step=1, total_steps=total_steps,
        step_name="Prétraitement des données"
    )

    preprocessor = Preprocessor(test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    X, y = preprocessor.prepare_data(df, target_col)
    column_types = preprocessor.identify_column_types(X)
    X_transformed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_transformed, y)

    st.session_state.preprocessor = preprocessor
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    # Tableaux récapitulatifs preprocessing
    try:
        preprocessing_summary = create_preprocessing_summary_table(X, X_transformed, preprocessor, column_types)
        split_summary = create_train_test_split_table(X_train, X_test, y_train, y_test)
        st.session_state.preprocessing_summary = preprocessing_summary
        st.session_state.split_summary = split_summary
        preprocessing_text = (
            "### ✅ Prétraitement terminé\n\n"
            "**Ce qui a été fait :**\n"
            "- séparation X / y\n"
            "- imputation valeurs manquantes\n"
            "- encodage catégoriel\n"
            "- normalisation\n"
            "- split train/test\n\n"
            "**Tableaux récapitulatifs ci-dessous.**\n\n"
            f"{_commentaire_dimensions(X, X_transformed)}\n"
            f"{_commentaire_split(X_train, X_test)}"
        )
        add_message(
            "assistant",
            preprocessing_text,
            content_type="mixed",
            text=preprocessing_text,
            dataframes=[preprocessing_summary, split_summary]
        )
    except Exception:
        pass

    add_message("assistant",
        f"### ✅ Prétraitement terminé !\n\n"
        f"**Dimensions** : {X.shape[1]} variable(s) → {X_transformed.shape[1]} après transformation\n"
        f"**Entraînement/Test** : {X_train.shape[0]}/{X_test.shape[0]} échantillons\n\n"
        f"{_commentaire_dimensions(X, X_transformed)}\n"
        f"{_commentaire_split(X_train, X_test)}"
    )

    # ============================================
    # ÉTAPE 2 : GESTION DU DÉSÉQUILIBRE
    # ============================================
    if task_type == 'classification':
        add_message("assistant",
            "<div style=\"font-size:1.6rem;font-weight:800;margin:0.25rem 0 0.5rem 0;\">⚖️ Étape 2/6 — Équilibrage des classes</div>"
            "<div style=\"margin:0 0 0.5rem 0;\"><b>Objectif</b> : détecter et corriger le déséquilibre</div>"
            "<div><b>Analyse en cours...</b></div>",
            content_type="progress",
            step=2, total_steps=total_steps,
            step_name="Analyse du déséquilibre"
        )

        resampler = Resampler(strategy=cfg.RESAMPLING_STRATEGY)
        is_imbalanced, dist_info = resampler.check_imbalance(y_train)

        if is_imbalanced:
            try:
                before_plot = plot_target_distribution(y_train, task_type)
            except Exception:
                before_plot = None
            X_train, y_train = resampler.apply_resampling(X_train, y_train)
            resampling_method = None
            try:
                if getattr(resampler, "resampler", None) is not None:
                    resampling_method = type(resampler.resampler).__name__
            except Exception:
                resampling_method = None
            try:
                after_plot = plot_target_distribution(y_train, task_type)
            except Exception:
                after_plot = None
            st.session_state.y_train = y_train
            strategy_requested = getattr(cfg, "RESAMPLING_STRATEGY", "auto")
            add_message("assistant",
                "### ⚠️ Déséquilibre corrigé\n\n"
                f"**Avant** : {dist_info['max_proportion']*100:.1f}% majoritaire\n"
                f"**Après** : Classes équilibrées\n\n"
                f"**Méthode utilisée** : `{resampling_method or 'inconnue'}` (stratégie demandée : `{strategy_requested}`)\n\n"
                "**Méthodes disponibles** : `smote`, `oversample`, `undersample`, `smote_tomek` (ou `auto`).\n\n"
                "Graphiques avant/après ci-dessous.\n\n"
                "Ce que j’ai trouvé : la distribution de la cible était déséquilibrée et a été corrigée pour stabiliser l’entraînement.",
                content_type="mixed",
                text=(
                    "### ⚠️ Déséquilibre corrigé\n\n"
                    f"**Avant** : {dist_info['max_proportion']*100:.1f}% majoritaire\n"
                    f"**Après** : Classes équilibrées\n\n"
                    f"**Méthode utilisée** : `{resampling_method or 'inconnue'}` (stratégie demandée : `{strategy_requested}`)\n\n"
                    "**Méthodes disponibles** : `smote`, `oversample`, `undersample`, `smote_tomek` (ou `auto`).\n\n"
                    "Graphiques avant/après ci-dessous.\n\n"
                    "Ce que j’ai trouvé : la distribution de la cible était déséquilibrée et a été corrigée pour stabiliser l’entraînement."
                ),
                plots=[p for p in [before_plot, after_plot] if p is not None]
            )
        else:
            add_message("assistant",
                "### ✅ Classes équilibrées\n\n"
                "Aucun rééchantillonnage nécessaire."
            )

    # ============================================
    # ÉTAPE 3 : RECHERCHE DE MODÈLES
    # ============================================
    add_message("assistant",
        "<div style=\"font-size:1.6rem;font-weight:800;margin:0.25rem 0 0.5rem 0;\">🔍 Étape 3/6 — Recherche de modèles</div>"
        "<div style=\"margin:0 0 0.5rem 0;\"><b>Objectif</b> : tester plusieurs modèles et hyperparamètres</div>"
        "<div><b>Recherche en cours...</b></div>",
        content_type="progress",
        step=3, total_steps=total_steps,
        step_name="Recherche de modèles"
    )

    searcher = ModelSearcher(
        cv=cfg.CV_FOLDS,
        scoring=None,
        search_method=cfg.SEARCH_METHOD,
        n_iter=cfg.N_ITER_RANDOM_SEARCH
    )
    search_results = searcher.search_models(X_train, y_train, task_type, metric)
    st.session_state.search_results = search_results

    add_message("assistant",
        f"### ✅ Recherche terminée !\n\n"
        f"**{len(search_results)} modèles** testés\n"
        f"**Meilleur score** : {max(r['best_score'] for r in search_results):.4f}\n\n"
        "Ce que j’ai trouvé : on compare plusieurs familles de modèles et on retient ceux qui obtiennent les meilleurs scores en validation."
    )

    # ============================================
    # ÉTAPE 4 : SÉLECTION DU MEILLEUR MODÈLE
    # ============================================
    add_message("assistant",
        "<div style=\"font-size:1.6rem;font-weight:800;margin:0.25rem 0 0.5rem 0;\">🏆 Étape 4/6 — Sélection du meilleur modèle</div>"
        "<div style=\"margin:0 0 0.5rem 0;\"><b>Objectif</b> : choisir le modèle le plus performant</div>"
        "<div><b>Sélection en cours...</b></div>",
        content_type="progress",
        step=4, total_steps=total_steps,
        step_name="Sélection du meilleur modèle"
    )

    selector = ModelSelector(task_type=task_type, metric=metric)
    selection_result = selector.select_best_model(search_results)
    st.session_state.selection_result = selection_result

    # Tableau + graphique de comparaison
    try:
        comparison_df = selection_result.get('comparison_summary')
        if comparison_df is not None:
            st.session_state.model_comparison_df = comparison_df
            score_col = 'Best Score' if 'Best Score' in comparison_df.columns else comparison_df.columns[1]
            comparison_plot = plot_model_comparison(comparison_df, score_col)
            selection_text = (
                "### 🏆 Modèle sélectionné\n\n"
                f"**{selection_result['best_model_name']}**\n\n"
                f"**Score** : {selection_result['best_score']:.4f}\n\n"
                "Tableau et graphique de comparaison ci-dessous.\n\n"
                f"{_commentaire_comparaison_modeles(comparison_df)}"
            )
            add_message(
                "assistant",
                selection_text,
                content_type="mixed",
                text=selection_text,
                plots=[comparison_plot] if comparison_plot else [],
                dataframes=[comparison_df]
            )
    except Exception:
        pass

    add_message("assistant",
        f"### 🏆 Modèle sélectionné !\n\n"
        f"**{selection_result['best_model_name']}**\n"
        f"**Score** : {selection_result['best_score']:.4f}"
    )

    # ============================================
    # ÉTAPE 5 : ÉVALUATION FINALE
    # ============================================
    add_message("assistant",
        "<div style=\"font-size:1.6rem;font-weight:800;margin:0.25rem 0 0.5rem 0;\">📊 Étape 5/6 — Évaluation finale</div>"
        "<div style=\"margin:0 0 0.5rem 0;\"><b>Objectif</b> : évaluer le modèle sur les données de test</div>"
        "<div><b>Évaluation en cours...</b></div>",
        content_type="progress",
        step=5, total_steps=total_steps,
        step_name="Évaluation du modèle"
    )

    evaluator = ModelEvaluator(task_type=task_type)
    evaluation = evaluator.evaluate_model(
        selection_result['best_model'], X_train, y_train, X_test, y_test, metric
    )
    st.session_state.evaluation = evaluation

    # Métriques + graphiques d'évaluation
    try:
        metrics_df = create_metrics_table(evaluation, task_type)
        plots = []
        train_test_plot = plot_train_test_comparison(evaluation, task_type)
        if train_test_plot:
            plots.append(train_test_plot)
        if task_type == 'classification' and evaluation.get('confusion_matrix'):
            cm_plot = plot_confusion_matrix(evaluation['confusion_matrix'])
            if cm_plot:
                plots.append(cm_plot)

        eval_text = (
            "### 📊 Évaluation terminée\n\n"
            "**Ce que nous vérifions :**\n"
            "- métriques sur Entraînement et Test\n"
            "- écart Entraînement/Test (sur-apprentissage / sous-apprentissage)\n"
            "- (classification) matrice de confusion\n\n"
            "Tableau et graphiques ci-dessous.\n\n"
            f"{_commentaire_train_test(evaluation, task_type)}"
        )
        add_message(
            "assistant",
            eval_text,
            content_type="mixed",
            text=eval_text,
            plots=plots,
            dataframes=[metrics_df] if metrics_df is not None else []
        )
    except Exception:
        pass

    add_message("assistant",
        "### 📊 Évaluation terminée !\n\n"
        "Résultats disponibles dans l'onglet **Résultats**."
    )

    # ============================================
    # ÉTAPE 6 : EXPLICATIONS
    # ============================================
    add_message("assistant",
        "<div style=\"font-size:1.6rem;font-weight:800;margin:0.25rem 0 0.5rem 0;\">💬 Étape 6/6 — Explications</div>"
        "<div style=\"margin:0 0 0.5rem 0;\"><b>Objectif</b> : générer des explications en langage naturel</div>"
        "<div><b>Analyse en cours...</b></div>",
        content_type="progress",
        step=6, total_steps=total_steps,
        step_name="Génération des explications"
    )

    explainer = st.session_state.llm_explainer
    explanation = explainer.explain_model_selection(selection_result, evaluation, task_type)

    try:
        suggestions = explainer.generate_improvement_suggestions(
            st.session_state.get('analysis') or {},
            evaluation,
            task_type
        )
    except Exception:
        suggestions = ""

    add_message("assistant",
        f"### 💡 Explication du modèle\n\n{explanation}\n\n{suggestions}"
    )

    st.success("✅ **Pipeline AutoML terminé avec succès !**")


if __name__ == "__main__":
    main()
