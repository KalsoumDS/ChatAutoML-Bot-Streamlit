"""
Interface Streamlit principale pour le ChatAutoML-Bot.
Met en œuvre une interface claire avec une barre latérale pour les contrôles
et une zone principale pour la conversation et les résultats.
Version avec un flux de travail étape par étape et des expanders dynamiques.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pathlib import Path
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.dashboard import create_final_summary
from typing import Any, Dict, List, Optional
from datetime import datetime
import io

from automl.data_loader import DataLoader
from automl.preprocessing import Preprocessor
from automl.evaluation import ModelEvaluator
from automl.selection import ModelSelector
from automl.search import ModelSearcher
from automl.resampling import Resampler
from llm.explainer import LLMExplainer
from app.visualizations import (
    plot_target_distribution, plot_missing_values,
    create_dataset_head_table, create_missing_values_table,
    create_descriptive_stats_table, create_metrics_table,
    plot_model_comparison, plot_feature_importance, plot_train_test_comparison
)
from app.chat_interface import detect_user_intent, generate_response


def _safe_filename(s: Optional[str]) -> str:
    name = (s or "document").strip()
    if not name:
        name = "document"
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)[:80]

# --- Configuration ---
try:
    import config.config as cfg
except ImportError:
    class DefaultConfig:
        MAX_FILE_SIZE_MB = 100; TEST_SIZE = 0.2; RANDOM_STATE = 42; CV_FOLDS = 5
        SEARCH_METHOD = 'grid'; N_ITER_RANDOM_SEARCH = 20; RESAMPLING_STRATEGY = 'auto'
    cfg = DefaultConfig()

CHATBOT_NAME = getattr(cfg, 'CHATBOT_NAME', "ChatAutoML-Bot")
CHATBOT_ICON = getattr(cfg, 'CHATBOT_ICON', '🤖')
CHATBOT_DESCRIPTION = getattr(cfg, 'CHATBOT_DESCRIPTION', 'Votre assistant AutoML pour données tabulaires.')

st.set_page_config(page_title=CHATBOT_NAME, page_icon=CHATBOT_ICON, layout="wide")

# --- Initialisation de la session ---
def initialize_session():
    """Initialise ou réinitialise l'état de la session."""
    llm_explainer = st.session_state.get('llm_explainer')
    if llm_explainer is None:
        llm_explainer = LLMExplainer(
            llm_provider=getattr(cfg, 'LLM_PROVIDER', 'template'),
            model_name=getattr(cfg, 'OLLAMA_MODEL', None),
            ollama_base_url=getattr(cfg, 'OLLAMA_BASE_URL', None),
        )
    st.session_state.clear()
    st.session_state.llm_explainer = llm_explainer
    st.session_state.messages = []
    st.session_state.dataset = None
    st.session_state.target_column = None
    st.session_state.task_type = None
    st.session_state.current_file_name = None
    st.session_state.analysis_done = False
    st.session_state.preprocessing_done = False
    st.session_state.automl_done = False

if 'messages' not in st.session_state:
    initialize_session()

# --- Fonctions Utilitaires ---
def add_message(role: str, content: Any, content_type: str = "text", **kwargs):
    st.session_state.messages.append({"role": role, "content": content, "content_type": content_type, **kwargs})

def _step_banner(step: int, total: int, title: str, subtitle: str = "") -> str:
    subtitle_html = f"<div style='opacity:.85;margin-top:.35rem;font-size:0.95rem'>{subtitle}</div>" if subtitle else ""
    return (
        "<div style='margin:0.35rem 0 0.75rem 0;padding:0.9rem 1rem;"
        "border-radius:14px;border:1px solid rgba(255,255,255,0.10);"
        "background:linear-gradient(135deg, rgba(59,130,246,0.18), rgba(147,51,234,0.12));'>"
        f"<div style='font-size:0.85rem;letter-spacing:.08em;text-transform:uppercase;opacity:.9;'>Étape {step}/{total}</div>"
        f"<div style='font-size:1.35rem;font-weight:900;margin-top:.15rem'>{title}</div>"
        f"{subtitle_html}"
        "</div>"
    )

def _resampling_method_explanation(method_name: Optional[str]) -> str:
    m = (method_name or "").lower()
    if not m:
        return "Méthode de rééchantillonnage non précisée."
    if "smote" in m and "tomek" in m:
        return "SMOTE-Tomek : génère des exemples synthétiques (SMOTE) puis nettoie les frontières (Tomek links)."
    if "smote" in m:
        return "SMOTE : crée des exemples synthétiques pour la classe minoritaire (pas juste des duplications)."
    if "randomoversampler" in m or "oversampler" in m:
        return "Oversampling : duplique aléatoirement des exemples de la classe minoritaire."
    if "randomundersampler" in m or "undersampler" in m:
        return "Undersampling : supprime aléatoirement des exemples de la classe majoritaire."
    return f"Méthode utilisée : {method_name}."

def _safe_float(x: Any):
    try:
        return float(x)
    except Exception:
        return None

def _findings_block(title: str, findings: list) -> str:
    items = "".join([f"<li style='margin:0.2rem 0'>{x}</li>" for x in findings if x])
    return (
        "<div style='margin:0.35rem 0 0.75rem 0;padding:0.9rem 1rem;"
        "border-radius:14px;border:1px solid rgba(255,255,255,0.10);"
        "background:rgba(255,255,255,0.03);'>"
        f"<div style='font-size:1.05rem;font-weight:900;margin-bottom:0.35rem'>{title}</div>"
        "<div style='opacity:.95'><b>Ce qu'on a trouvé :</b></div>"
        f"<ul style='margin:0.35rem 0 0 1.1rem;padding:0;opacity:.95'>{items}</ul>"
        "</div>"
    )

def _comment_block(comment: str) -> str:
    if not comment:
        return ""
    return (
        "<div style='margin:-0.35rem 0 0.75rem 0;padding:0.7rem 1rem;"
        "border-radius:14px;border:1px dashed rgba(255,255,255,0.12);"
        "background:rgba(255,255,255,0.02);'>"
        "<div style='font-weight:900;margin-bottom:0.25rem'>📝 Commentaire</div>"
        f"<div style='opacity:.92'>{comment}</div>"
        "</div>"
    )

def _reset_pipeline_state(keep_messages: bool = True):
    if not keep_messages:
        st.session_state.messages = []
    st.session_state.target_column = None
    st.session_state.task_type = None
    st.session_state.analysis_done = False
    st.session_state.preprocessing_done = False
    st.session_state.automl_done = False
    for k in [
        'analysis', 'preprocessor',
        'X_train', 'X_test', 'y_train', 'y_test',
        'search_results', 'selection_result', 'evaluation', 'feature_importance',
        'model_comparison_df', 'analysis_info', 'preprocessing_info', 'resampling_info',
    ]:
        if k in st.session_state:
            del st.session_state[k]

def _clear_after(stage: str):
    if stage == 'analysis':
        st.session_state.analysis_done = False
        st.session_state.preprocessing_done = False
        st.session_state.automl_done = False
        for k in ['analysis', 'analysis_info', 'preprocessor', 'X_train', 'X_test', 'y_train', 'y_test',
                  'preprocessing_info', 'search_results', 'selection_result', 'evaluation',
                  'model_comparison_df', 'resampling_info']:
            if k in st.session_state:
                del st.session_state[k]
        return
    if stage == 'preprocessing':
        st.session_state.preprocessing_done = False
        st.session_state.automl_done = False
        for k in ['preprocessor', 'X_train', 'X_test', 'y_train', 'y_test', 'preprocessing_info',
                  'search_results', 'selection_result', 'evaluation', 'model_comparison_df', 'resampling_info']:
            if k in st.session_state:
                del st.session_state[k]
        return
    if stage == 'automl':
        st.session_state.automl_done = False
        for k in ['search_results', 'selection_result', 'evaluation', 'model_comparison_df', 'resampling_info']:
            if k in st.session_state:
                del st.session_state[k]
        return

def _handle_pending_chat_action(user_text: str) -> bool:
    pending = st.session_state.get('pending_chat_action')
    if not pending:
        return False
    t = (user_text or "").strip().lower()
    if pending == 'confirm_reset_session':
        if t in {'oui', 'yes', 'y', 'ok', "d'accord", "daccord"}:
            initialize_session()
            add_message("assistant", "✅ Session réinitialisée. Vous pouvez recharger un dataset.")
            st.session_state.pending_chat_action = None
            return True
        if t in {'non', 'no', 'n'}:
            add_message("assistant", "D'accord. Je ne réinitialise pas la session.")
            st.session_state.pending_chat_action = None
            return True
        add_message("assistant", "Je te laisse confirmer : réponds `oui` ou `non`.")
        return True
    if pending == 'choose_reset_scope':
        if t in {'a', 'option a'}:
            _reset_pipeline_state(keep_messages=True)
            add_message("assistant", "✅ OK. J'ai réinitialisé le pipeline. Le dataset est conservé.")
            st.session_state.pending_chat_action = None
            return True
        if t in {'b', 'option b'}:
            initialize_session()
            add_message("assistant", "✅ OK. J'ai tout réinitialisé (dataset + cible + résultats + chat).")
            st.session_state.pending_chat_action = None
            return True
        add_message("assistant", "Je te laisse choisir : réponds `A` (refaire pipeline) ou `B` (tout réinitialiser).")
        return True
    return False

@st.cache_data(show_spinner=False)
def _load_tabular_data_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return _load_tabular_data(io.BytesIO(file_bytes), filename)

def _load_tabular_data(source: io.BytesIO, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    source.seek(0)
    if ext == '.csv':
        try:
            sample = source.read(4096).decode('utf-8', errors='ignore')
            import csv
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            sep = dialect.delimiter
        except Exception:
            seps = [',', ';', '\t']
            counts = {s: sample.count(s) for s in seps}
            sep = max(counts, key=counts.get) if any(counts.values()) else ','
        source.seek(0)
        return pd.read_csv(source, sep=sep)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(source)
    elif ext == '.parquet':
        return pd.read_parquet(source)
    elif ext == '.json':
        return pd.read_json(source)
    else:
        raise ValueError(f"Format de fichier non supporté : {ext}")

def display_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=CHATBOT_ICON if msg["role"] == "assistant" else "👤"):
            content = msg.get("content")
            if msg.get("content_type", "text") == "text":
                st.markdown(content, unsafe_allow_html=True)
            elif msg.get("content_type") == "dataframe":
                st.dataframe(content, width='stretch')
            elif msg.get("content_type") == "plot":
                st.plotly_chart(content, width='stretch')
            elif msg.get("content_type") == "mixed":
                if "text" in content:
                    st.markdown(content["text"], unsafe_allow_html=True)
                if "markdown" in content and content.get("markdown"):
                    st.markdown(content["markdown"])
                if "dataframes" in content:
                    for df in content.get("dataframes", []):
                        if df is not None:
                            st.dataframe(df, width='stretch')
                if "plots" in content:
                    for plot in content.get("plots", []):
                        if plot:
                            st.plotly_chart(plot, width='stretch')

# --- Fonctions du Pipeline ---

def perform_analysis():
    df = st.session_state.dataset
    target = st.session_state.target_column
    add_message("assistant", _step_banner(1, 6, "🔎 Analyse initiale du dataset",
        "Objectif : comprendre la structure (taille, valeurs manquantes, statistiques, cible)."))

    try:
        n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
        shape_result = f"df.shape = <b>({n_rows}, {n_cols})</b>"
    except Exception:
        n_rows, n_cols = None, None
        shape_result = "df.shape disponible"

    try:
        missing_total = int(df.isna().sum().sum())
        missing_pct_total = float(missing_total / max(1, df.size) * 100)
    except Exception:
        missing_total, missing_pct_total = None, None

    try:
        dup_count = int(df.duplicated().sum())
    except Exception:
        dup_count = None

    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        bool_cols = [c for c in num_cols if df[c].nunique(dropna=True) <= 2]
        numeric_cols = [c for c in num_cols if c not in bool_cols]
        cat_cols = [c for c in df.columns if c not in num_cols]
    except Exception:
        numeric_cols, bool_cols, cat_cols = [], [], []

    # ✅ FIX: tous les éléments en str pour éviter ArrowTypeError
    summary_df = pd.DataFrame(
        {
            'Valeur': [
                f"({df.shape[0]}, {df.shape[1]})" if hasattr(df, 'shape') else "N/A",
                str(missing_total) if missing_total is not None else "N/A",
                f"{missing_pct_total:.2f}%" if missing_pct_total is not None else "N/A",
                str(dup_count) if dup_count is not None else "N/A",
                str(len(numeric_cols)),
                str(len(cat_cols)),
                str(len(bool_cols)),
            ]
        },
        index=['df.shape', 'Valeurs manquantes (total)', 'Valeurs manquantes (%)',
               'Lignes dupliquées', 'Colonnes numériques', 'Colonnes catégorielles/texte', 'Colonnes booléennes'],
    )

    add_message("assistant", content={
        "text": _findings_block("📌 Résumé dataset", [
            shape_result,
            f"Valeurs manquantes total: <b>{missing_total}</b> (≈ <b>{missing_pct_total:.2f}%</b>)" if missing_total is not None and missing_pct_total is not None else "Valeurs manquantes calculées.",
            f"Lignes dupliquées: <b>{dup_count}</b>" if dup_count is not None else "Doublons calculés.",
        ]) + _comment_block(
            f"Taille : {n_rows}×{n_cols}. "
            + (f"Manquants : {missing_pct_total:.2f}% global. " if missing_pct_total is not None else "")
            + (f"Doublons : {dup_count}." if dup_count is not None else "")
        ),
        "dataframes": [summary_df],
    }, content_type="mixed")

    # analysis_info pour résumé final
    try:
        analysis_info = {
            'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
            'target_info': {'column': target},
            'columns': {}, 'missing_values': {},
        }
        for c in df.columns:
            t = 'numeric' if pd.api.types.is_numeric_dtype(df[c]) else 'categorical'
            analysis_info['columns'][c] = {'type': t}
        missing_counts = df.isna().sum()
        for c in df.columns:
            cnt = int(missing_counts.get(c, 0))
            analysis_info['missing_values'][c] = {'count': cnt, 'percentage': float(cnt / max(1, len(df)) * 100)}
        st.session_state.analysis_info = analysis_info
    except Exception:
        pass

    head_df = create_dataset_head_table(df)
    stats_df = create_descriptive_stats_table(df)

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in numeric_cols]
    except Exception:
        numeric_cols, cat_cols = [], []

    suspected_numeric_text_cols = []
    try:
        for c in cat_cols[:50]:
            s = df[c].dropna()
            if s.empty:
                continue
            sample = s.astype(str).head(2000)
            conv = pd.to_numeric(sample.str.replace(',', '.', regex=False), errors='coerce')
            ratio = float(conv.notna().mean()) if len(sample) else 0.0
            if ratio >= 0.9:
                suspected_numeric_text_cols.append(c)
            if len(suspected_numeric_text_cols) >= 5:
                break
    except Exception:
        suspected_numeric_text_cols = []

    head_result_lines = []
    if n_rows is not None and n_cols is not None:
        head_result_lines.append(f"- Lignes × colonnes : <b>{n_rows}</b> × <b>{n_cols}</b>")
    head_result_lines.append(f"- Colonnes numériques détectées : <b>{len(numeric_cols)}</b>")
    head_result_lines.append(f"- Colonnes non-numériques (texte/cat.) : <b>{len(cat_cols)}</b>")
    if dup_count is not None:
        head_result_lines.append(f"- Lignes dupliquées : <b>{dup_count}</b>")
    if suspected_numeric_text_cols:
        head_result_lines.append("- Colonnes potentiellement numériques mais stockées en texte : "
            + ", ".join([f"<code>{c}</code>" for c in suspected_numeric_text_cols]))

    add_message("assistant", content={
        "text": _findings_block("📌 Aperçu des données (head)", head_result_lines if head_result_lines else ["Aperçu généré."])
            + _comment_block(("Colonnes suspectes num-en-texte détectées : " + ", ".join(suspected_numeric_text_cols))
                if suspected_numeric_text_cols else "Aucune colonne num-en-texte détectée sur l'échantillon analysé."),
        "dataframes": [head_df],
    }, content_type="mixed")

    stats_extra_comment = ""
    try:
        num_list = df.select_dtypes(include=[np.number]).columns.tolist()
        probe_cols = num_list[:20]
        if probe_cols:
            rng = (df[probe_cols].max(numeric_only=True) - df[probe_cols].min(numeric_only=True)).sort_values(ascending=False)
            if not rng.empty:
                top_col = str(rng.index[0])
                top_rng = _safe_float(rng.iloc[0])
                if top_rng is not None:
                    stats_extra_comment = f"Plus grande amplitude observée : <code>{top_col}</code> (range ≈ {top_rng:.3g})."
    except Exception:
        stats_extra_comment = ""

    add_message("assistant", content={
        "text": _findings_block("📊 Statistiques descriptives", [
            "Résumé statistique des colonnes numériques (table ci-dessous).",
            f"Numériques : <b>{len(numeric_cols)}</b> — Catégorielles/texte : <b>{len(cat_cols)}</b>.",
        ]) + _comment_block((stats_extra_comment + " " if stats_extra_comment else "")
            + "Min/Max et échelle expliquent pourquoi on applique un scaling pour certains modèles (SVM/KNN/régression)."),
        "dataframes": [stats_df],
    }, content_type="mixed")

    missing_plot = plot_missing_values(df)
    try:
        missing_counts = df.isna().sum().sort_values(ascending=False)
        missing_pct = (missing_counts / max(1, len(df)) * 100).round(2)
        missing_df = pd.DataFrame({'missing_count': missing_counts.astype(int), 'missing_pct': missing_pct})
        total_missing = int(missing_counts.sum())
        pct_missing_total = float(total_missing / max(1, df.size) * 100)
        top = missing_df[missing_df['missing_count'] > 0].head(3)
        if not top.empty:
            top_str = ", ".join([f"{idx}: {int(row['missing_count'])} ({float(row['missing_pct']):.2f}%)" for idx, row in top.iterrows()])
            miss_comment = f"df.isnull().sum() calculé. Total : <b>{total_missing}</b> (≈ <b>{pct_missing_total:.2f}%</b>).<br/>Colonnes les plus touchées : <b>{top_str}</b>."
        else:
            miss_comment = f"Aucune valeur manquante détectée (total = <b>{total_missing}</b>)."
    except Exception:
        miss_comment = "Analyse des valeurs manquantes calculée."
        missing_df = create_missing_values_table(df)

    add_message("assistant", content={
        "text": _findings_block("🧩 Valeurs manquantes", [
            "df.isnull().sum() + % par colonne (table ci-dessous) + graphique.", miss_comment,
        ]) + _comment_block("Imputation prévue : numériques→médiane, catégorielles→'Unknown', bool→mode."),
        "dataframes": [missing_df],
        "plots": [missing_plot] if missing_plot is not None else [],
    }, content_type="mixed")

    if target:
        target_dist_plot = plot_target_distribution(df[target], st.session_state.task_type)
        try:
            y = df[target]
            if st.session_state.task_type == 'classification':
                counts = y.value_counts(dropna=False)
                major_pct = float(counts.max() / max(1, counts.sum()) * 100)
                tgt_comment = f"Classes : <b>{int(counts.shape[0])}</b> — Majoritaire : <b>{major_pct:.1f}%</b>."
            else:
                tgt_comment = "On observe la forme (asymétrie) et les outliers."
        except Exception:
            tgt_comment = "Distribution calculée."

        target_counts_df = None
        try:
            vc = df[target].value_counts(dropna=False)
            vp = (vc / max(1, vc.sum()) * 100).round(2)
            target_counts_df = pd.DataFrame({'count': vc.astype(int), 'pct': vp})
        except Exception:
            target_counts_df = None

        tgt_specific_comment = ""
        try:
            if st.session_state.task_type == 'classification' and target_counts_df is not None and not target_counts_df.empty:
                major_label = str(target_counts_df['count'].idxmax())
                major_pct2 = _safe_float(target_counts_df.loc[target_counts_df['count'].idxmax(), 'pct'])
                major_cnt = int(target_counts_df.loc[target_counts_df['count'].idxmax(), 'count'])
                tgt_specific_comment = (f"Classe majoritaire = <code>{major_label}</code> : {major_cnt} ({major_pct2:.2f}%). "
                    + ("Dataset déséquilibré → rééchantillonnage appliqué dans AutoML." if major_pct2 is not None and major_pct2 >= 70 else "Dataset plutôt équilibré."))
        except Exception:
            tgt_specific_comment = ""

        add_message("assistant", content={
            "text": _findings_block(f"🎯 Distribution de la cible ({target})", [tgt_comment])
                + _comment_block(tgt_specific_comment or "Distribution calculée (voir table + graphe)."),
            "dataframes": [target_counts_df] if target_counts_df is not None else [],
            "plots": [target_dist_plot] if target_dist_plot is not None else [],
        }, content_type="mixed")

    add_message("assistant", "✅ Analyse terminée. Vous pouvez maintenant lancer le prétraitement.")
    st.session_state.analysis_done = True

def perform_preprocessing():
    df = st.session_state.dataset
    target_col = st.session_state.target_column
    add_message("assistant", _step_banner(2, 6, "🧼 Prétraitement (nettoyage + encodage + scaling)",
        "Objectif : convertir le dataset en matrice numérique propre, prête pour l'entraînement."))

    preprocessor = Preprocessor(test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    X, y = preprocessor.prepare_data(df, target_col)

    try:
        col_types = preprocessor.identify_column_types(X)
        n_num = len(col_types.get('numeric', []))
        n_cat = len(col_types.get('categorical', []))
        n_bool = len(col_types.get('boolean', []))
        diag_result = f"Numériques : <b>{n_num}</b><br/>Catégorielles : <b>{n_cat}</b><br/>Booléennes : <b>{n_bool}</b>"
    except Exception:
        diag_result = "Types de colonnes détectés automatiquement."

    add_message("assistant", content={
        "text": _findings_block("🔎 Diagnostic des colonnes", [
            diag_result, "Traitement appliqué : num (imputation+scaling), cat (imputation+onehot), bool (imputation).",
        ]) + _comment_block("On prépare une matrice 100% numérique et sans NaN pour que tous les modèles puissent s'entraîner correctement."),
    }, content_type="mixed")

    X_transformed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_transformed, y)
    feature_names = getattr(preprocessor, 'feature_names', None)
    feature_names_list = list(feature_names) if feature_names is not None else []
    st.session_state.update(preprocessor=preprocessor, feature_names=feature_names_list,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # ✅ FIX: tous les éléments en str
    preprocess_summary_df = pd.DataFrame(
        {'Valeur': [str(int(X.shape[1])), str(int(X_transformed.shape[1])),
                    str(int(X_train.shape[0])), str(int(X_test.shape[0]))]},
        index=['X avant (colonnes)', 'X après (features)', 'Train (lignes)', 'Test (lignes)'],
    )

    cat_card_df = None
    try:
        col_types = preprocessor.identify_column_types(X)
        cat_cols = col_types.get('categorical', [])
        if cat_cols:
            card = X[cat_cols].nunique(dropna=True).sort_values(ascending=False).head(10)
            cat_card_df = pd.DataFrame({'n_unique': card.astype(int)})
    except Exception:
        cat_card_df = None

    feature_sample_df = None
    try:
        sample = feature_names_list[:60]
        if sample:
            feature_sample_df = pd.DataFrame({'feature_name': [str(s)[:80] for s in sample]})
    except Exception:
        feature_sample_df = None

    transformed_preview_df = None
    try:
        n_rows_preview = min(5, int(X_transformed.shape[0]))
        n_cols_preview = min(10, int(X_transformed.shape[1]))
        cols_preview = feature_names_list[:n_cols_preview] if feature_names_list else [f"f{i}" for i in range(n_cols_preview)]
        X_slice = X_transformed[:n_rows_preview, :n_cols_preview]
        if hasattr(X_slice, "toarray"):
            X_slice = X_slice.toarray()
        transformed_preview_df = pd.DataFrame(X_slice, columns=[str(c)[:40] for c in cols_preview])
    except Exception:
        transformed_preview_df = None

    add_message("assistant", content={
        "text": _findings_block("✅ Résultat du prétraitement", [
            "Résumé chiffré + tables ci-dessous.",
            "(On évite d'afficher les features en texte : elles sont listées dans un tableau, tronquées.)",
        ]) + _comment_block(
            (f"Cardinalité max détectée : <code>{str(cat_card_df.index[0])}</code> = {int(cat_card_df.iloc[0,0])} valeurs uniques. " if cat_card_df is not None and not cat_card_df.empty else "")
            + f"Expansion OneHot : {int(X.shape[1])} → {int(X_transformed.shape[1])} features."
        ),
        "dataframes": [preprocess_summary_df, cat_card_df, feature_sample_df, transformed_preview_df],
    }, content_type="mixed")

    try:
        preprocessing_info = {
            'n_features_before': int(X.shape[1]), 'n_features_after': int(X_transformed.shape[1]),
            'n_train': int(X_train.shape[0]), 'n_test': int(X_test.shape[0]),
            'train_pct': round(float(X_train.shape[0] / max(1, X_train.shape[0] + X_test.shape[0]) * 100), 2),
            'test_pct': round(float(X_test.shape[0] / max(1, X_train.shape[0] + X_test.shape[0]) * 100), 2),
            'cv_folds': int(st.session_state.get('cv_folds', getattr(cfg, 'CV_FOLDS', 5))),
            'search_method': st.session_state.get('search_method', getattr(cfg, 'SEARCH_METHOD', 'grid')),
            'metric': st.session_state.get('metric', 'auto'),
        }
        st.session_state.preprocessing_info = preprocessing_info
    except Exception:
        pass
    st.session_state.preprocessing_done = True

def perform_automl():
    X_train, y_train = st.session_state.X_train, st.session_state.y_train
    X_test, y_test = st.session_state.X_test, st.session_state.y_test
    task_type = st.session_state.task_type
    metric = st.session_state.get('metric', 'auto')
    cv_folds = int(st.session_state.get('cv_folds', getattr(cfg, 'CV_FOLDS', 5)))
    search_method = st.session_state.get('search_method', getattr(cfg, 'SEARCH_METHOD', 'grid'))
    n_iter = int(st.session_state.get('n_iter', getattr(cfg, 'N_ITER_RANDOM_SEARCH', 20)))
    resampling_strategy = st.session_state.get('resampling_strategy', getattr(cfg, 'RESAMPLING_STRATEGY', 'auto'))

    add_message("assistant",
        _step_banner(3, 6, "🤖 AutoML (recherche + sélection + évaluation)",
            "Objectif : tester plusieurs modèles, optimiser des hyperparamètres, puis sélectionner le meilleur.")
        + "<div style='margin-top:-0.25rem'>"
        + "<div style='font-weight:800;margin:0.1rem 0 0.35rem 0'>⚙️ Configuration</div>"
        + f"<div style='opacity:.9'>- <b>Type</b> : <code>{task_type}</code></div>"
        + f"<div style='opacity:.9'>- <b>Métrique</b> : <code>{metric}</code></div>"
        + f"<div style='opacity:.9'>- <b>CV</b> : <code>{cv_folds}</code>-fold</div>"
        + f"<div style='opacity:.9'>- <b>Recherche</b> : <code>{search_method}</code>"
          + (f" (n_iter={n_iter})" if search_method == 'random' else "") + "</div>"
        + (f"<div style='opacity:.9'>- <b>Déséquilibre</b> : <code>{resampling_strategy}</code></div>" if task_type == 'classification' else "")
        + "</div>")

    with st.status("🚀 Pipeline AutoML en cours...", expanded=True) as status:
        try:
            if task_type == 'classification':
                status.write("⚖️ Gestion du déséquilibre...")
                resampler = Resampler(strategy=resampling_strategy)
                is_imbalanced, dist_info = resampler.check_imbalance(y_train)
                if is_imbalanced:
                    y_before = y_train.copy()
                    try:
                        before_plot = plot_target_distribution(y_before, task_type)
                    except Exception:
                        before_plot = None
                    X_train, y_train = resampler.apply_resampling(X_train, y_train)
                    y_after = y_train
                    try:
                        after_plot = plot_target_distribution(y_after, task_type)
                    except Exception:
                        after_plot = None

                    imbalance_df = None
                    try:
                        before_counts = y_before.value_counts(dropna=False)
                        after_counts = y_after.value_counts(dropna=False)
                        classes = sorted(set(before_counts.index.tolist()) | set(after_counts.index.tolist()), key=lambda x: str(x))
                        imbalance_df = pd.DataFrame({
                            'before_count': [int(before_counts.get(c, 0)) for c in classes],
                            'before_pct': [round(float(before_counts.get(c, 0) / max(1, before_counts.sum()) * 100), 2) for c in classes],
                            'after_count': [int(after_counts.get(c, 0)) for c in classes],
                            'after_pct': [round(float(after_counts.get(c, 0) / max(1, after_counts.sum()) * 100), 2) for c in classes],
                        }, index=[str(c) for c in classes])
                    except Exception:
                        imbalance_df = None

                    try:
                        resampling_method = type(getattr(resampler, 'resampler', None)).__name__ if getattr(resampler, 'resampler', None) is not None else None
                    except Exception:
                        resampling_method = None

                    add_message("assistant", content={
                        "text": _findings_block("⚖️ Déséquilibre des classes (avant / après)", [
                            f"Majoritaire avant : <b>{dist_info.get('max_proportion', 0)*100:.1f}%</b>",
                            f"Stratégie demandée : <code>{resampling_strategy}</code>",
                            f"Méthode utilisée : <code>{resampling_method or 'inconnue'}</code>",
                        ]) + _comment_block(_resampling_method_explanation(resampling_method)
                            + f" Avant : {int(len(y_before))} lignes, après : {int(len(y_after))} lignes."),
                        "dataframes": [df for df in [imbalance_df] if df is not None],
                        "plots": [p for p in [before_plot, after_plot] if p is not None],
                    }, content_type="mixed")

                    try:
                        st.session_state.resampling_info = {
                            'applied': True,
                            'initial_distribution': str(dict(before_counts)) if 'before_counts' in locals() else None,
                            'strategy': resampling_strategy,
                            'n_before': int(len(y_before)), 'n_after': int(len(y_after)),
                            'method': resampling_method,
                        }
                    except Exception:
                        pass
                else:
                    add_message("assistant", "✅ Classes équilibrées : aucun rééchantillonnage nécessaire.")
                    st.session_state.resampling_info = {'applied': False}

            status.write("🔍 Recherche de modèles...")
            searcher = ModelSearcher(cv=cv_folds, search_method=search_method, n_iter=n_iter)
            search_results = searcher.search_models(X_train, y_train, task_type, metric)
            st.session_state.search_results = search_results
            add_message("assistant", _step_banner(4, 6, "🔍 Recherche de modèles terminée",
                f"{len(search_results)} modèles testés (CV : moyenne + écart-type sur plusieurs métriques)."))

            status.write("🏆 Sélection du meilleur modèle...")
            selector = ModelSelector(task_type=task_type, metric=metric)
            selection_result = selector.select_best_model(search_results)
            st.session_state.selection_result = selection_result
            add_message("assistant",
                _step_banner(5, 6, "🏆 Sélection du meilleur modèle",
                    f"Retenu : {selection_result['best_model_name']} — score CV : {selection_result['best_score']:.4f}")
                + "Commentaire : ce choix est fait sur la <b>validation croisée</b> pour réduire le risque de sur-apprentissage.")

            try:
                comparison_df = selection_result.get('comparison_summary')
                if comparison_df is not None:
                    model_comment = ""
                    try:
                        if 'Best Score' in comparison_df.columns and 'Model' in comparison_df.columns:
                            tmp = comparison_df[['Model', 'Best Score']].dropna().sort_values('Best Score', ascending=False)
                            if not tmp.empty:
                                best_model_name = str(tmp.iloc[0]['Model'])
                                best_score = _safe_float(tmp.iloc[0]['Best Score'])
                                model_comment = f"Meilleur = <code>{best_model_name}</code> (Best Score = {best_score:.4f})."
                                if len(tmp) > 1:
                                    second_score = _safe_float(tmp.iloc[1]['Best Score'])
                                    if second_score is not None and best_score is not None:
                                        model_comment += f" Écart au 2e ≈ {best_score-second_score:+.4f}."
                    except Exception:
                        model_comment = ""
                    st.session_state.model_comparison_df = comparison_df
                    comp_plot = plot_model_comparison(comparison_df, metric='Best Score')
                    add_message("assistant", content={
                        "text": _findings_block("📋 Comparaison des modèles", [
                            "Tableau + graphique des scores (validation croisée).",
                        ]) + _comment_block(model_comment or "Comparer Best Score + temps + stabilité."),
                        "dataframes": [comparison_df],
                        "plots": [comp_plot] if comp_plot is not None else [],
                    }, content_type="mixed")
            except Exception:
                pass

            status.write("📊 Évaluation finale...")
            evaluator = ModelEvaluator(task_type=task_type)
            evaluation = evaluator.evaluate_model(selection_result['best_model'], X_train, y_train, X_test, y_test, metric)
            st.session_state.evaluation = evaluation
            metrics_df = create_metrics_table(evaluation, task_type)

            plots = []
            try:
                train_test_plot = plot_train_test_comparison(evaluation, task_type)
                if train_test_plot is not None:
                    plots.append(train_test_plot)
            except Exception:
                pass

            diag_lines = []
            try:
                if evaluation.get('overfitting', {}).get('detected'):
                    gap = evaluation['overfitting'].get('gap')
                    diag_lines.append(f"⚠️ <b>Overfitting détecté</b> (écart train/test ≈ {gap:.3f}).")
                elif evaluation.get('underfitting', {}).get('detected'):
                    diag_lines.append("⚠️ <b>Underfitting détecté</b> (scores faibles sur train et test).")
                else:
                    diag_lines.append("✅ <b>Généralisation correcte</b> (écart train/test acceptable).")
            except Exception:
                pass

            add_message("assistant", content={
                "text": _step_banner(6, 6, "📊 Évaluation finale", "Train vs test + diagnostic de généralisation")
                    + _findings_block("📊 Évaluation finale (train vs test)",
                        ["<br/>".join(diag_lines) if diag_lines else "Diagnostic de généralisation effectué."]),
                "dataframes": [metrics_df] if metrics_df is not None else [],
                "plots": plots,
            }, content_type="mixed")

            try:
                feature_names = st.session_state.get('feature_names')
                if feature_names:
                    fi = evaluator.get_feature_importance(selection_result['best_model'], feature_names)
                    if fi:
                        fi_top_comment = ""
                        try:
                            top_items = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:3]
                            if top_items:
                                fi_top_comment = "Top : " + ", ".join([f"<code>{k}</code> ({_safe_float(v):.3g})" for k, v in top_items])
                        except Exception:
                            fi_top_comment = ""
                        fi_plot = plot_feature_importance(fi, top_n=15)
                        add_message("assistant", content={
                            "text": _findings_block("💡 Importance des variables", ["Graphique des variables les plus influentes."])
                                + _comment_block(fi_top_comment or "Les premières variables du graphe sont celles qui contribuent le plus."),
                            "plots": [fi_plot] if fi_plot is not None else [],
                        }, content_type="mixed")
            except Exception:
                pass

            try:
                summary_md = create_final_summary(
                    analysis=st.session_state.get('analysis_info') or {},
                    preprocessing_info=st.session_state.get('preprocessing_info') or {},
                    resampling_info=st.session_state.get('resampling_info'),
                    search_results=search_results,
                    selection_result=selection_result,
                    evaluation=evaluation,
                    task_type=task_type,
                )
                add_message("assistant", content={
                    "text": _findings_block("🧾 Résumé global AutoML", ["Rapport complet ci-dessous."]),
                    "markdown": summary_md,
                }, content_type="mixed")
            except Exception:
                pass

            status.update(label="✅ Pipeline AutoML terminé !", state="complete")
            st.session_state.automl_done = True
            st.balloons()
        except Exception as e:
            status.update(label="❌ Erreur durant le pipeline", state="error")
            add_message("assistant", f"❌ <b>Erreur :</b> {e}")

# --- Interface principale ---
def main():
    st.title(f"{CHATBOT_ICON} {CHATBOT_NAME}")

    dataset_loaded = st.session_state.dataset is not None
    analysis_done = st.session_state.get('analysis_done', False)
    preprocessing_done = st.session_state.get('preprocessing_done', False)
    automl_done = st.session_state.get('automl_done', False)

    with st.sidebar:
        st.header("Panneau de Contrôle")

        with st.expander("1. Charger un Dataset", expanded=not dataset_loaded):
            uploaded_file = st.file_uploader("Importer un fichier",
                type=['csv', 'xlsx', 'xls', 'parquet', 'json'], label_visibility="collapsed")
            if uploaded_file:
                file_sig = (uploaded_file.name, getattr(uploaded_file, 'size', None))
                if file_sig != st.session_state.get('uploaded_file_sig'):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        df = _load_tabular_data_cached(file_bytes, uploaded_file.name)
                        st.session_state.dataset = df
                        st.session_state.current_file_name = uploaded_file.name
                        st.session_state.uploaded_file_sig = file_sig
                        _reset_pipeline_state(keep_messages=True)
                        add_message("assistant", f"✅ Dataset `{uploaded_file.name}` chargé. Choisissez une cible.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur de lecture: {e}")

        if dataset_loaded:
            st.success(f"Dataset: `{st.session_state.current_file_name}`")

        if dataset_loaded:
            with st.expander("2. Analyser les Données", expanded=dataset_loaded and not analysis_done):
                df = st.session_state.dataset
                target_options = ["-- Sélectionner --"] + list(df.columns)
                current_target_idx = target_options.index(st.session_state.target_column) if st.session_state.get('target_column') in target_options else 0
                selected_target = st.selectbox("Colonne Cible", target_options, index=current_target_idx, key="target_selectbox")

                if selected_target != "-- Sélectionner --" and selected_target != st.session_state.get('target_column'):
                    st.session_state.target_column = selected_target
                    st.session_state.task_type = DataLoader().detect_task_type(df, selected_target)
                    add_message("assistant", f"🎯 Cible : `{selected_target}` (Tâche : {st.session_state.task_type}).")
                    st.rerun()

                if st.session_state.get('target_column') and not analysis_done:
                    if st.button("🔍 Analyser le Dataset", use_container_width=True, type="primary"):
                        perform_analysis()
                        st.rerun()
            if analysis_done:
                st.success("Analyse terminée.")

        if analysis_done:
            with st.expander("3. Prétraitement", expanded=analysis_done and not preprocessing_done):
                if not preprocessing_done:
                    if st.button("🔧 Lancer le Prétraitement", use_container_width=True, type="primary"):
                        perform_preprocessing()
                        st.rerun()
            if preprocessing_done:
                st.success("Prétraitement terminé.")

        if preprocessing_done:
            with st.expander("4. Lancer AutoML", expanded=preprocessing_done and not automl_done):
                if not automl_done:
                    if st.button("🚀 Lancer l'AutoML", use_container_width=True, type="primary"):
                        perform_automl()
                        st.rerun()
            if automl_done:
                st.success("AutoML terminé.")

        st.divider()
        if st.button("🔄 Recommencer", use_container_width=True):
            initialize_session()
            st.rerun()

    with st.container():
        if not st.session_state.messages:
            add_message("assistant", "Bonjour ! Chargez un dataset pour commencer.")
        display_chat()

    if prompt := st.chat_input("Posez une question..."):
        add_message("user", prompt)
        if _handle_pending_chat_action(prompt):
            st.rerun()

        context = {k: st.session_state.get(k) for k in [
            'task_type', 'selection_result', 'evaluation', 'dataset',
            'target_column', 'analysis_done', 'preprocessing_done', 'automl_done',
        ]}
        try:
            intent = detect_user_intent(prompt)
            intent_type = intent.get('intent', 'general')

            if intent_type == 'reset_pipeline':
                st.session_state.pending_chat_action = 'choose_reset_scope'
                add_message("assistant", generate_response(intent=intent, context=context, llm_explainer=st.session_state.llm_explainer, message=prompt))
                st.rerun()
            if intent_type == 'reset_session':
                st.session_state.pending_chat_action = 'confirm_reset_session'
                add_message("assistant", generate_response(intent=intent, context=context, llm_explainer=st.session_state.llm_explainer, message=prompt))
                st.rerun()
            if intent_type == 'explain_step':
                add_message("assistant", generate_response(intent=intent, context=context, llm_explainer=st.session_state.llm_explainer, message=prompt))
                st.rerun()
            if intent_type == 'redo_step':
                add_message("assistant", generate_response(intent=intent, context=context, llm_explainer=st.session_state.llm_explainer, message=prompt))
                entities = intent.get('entities') or {}
                section = entities.get('section')
                step = entities.get('step')
                if step in (1,): section = 'analysis'
                elif step in (2,): section = 'preprocessing'
                elif step in (3, 4, 5, 6): section = 'automl'
                if section == 'analysis':
                    _clear_after('analysis'); perform_analysis(); st.rerun()
                if section == 'preprocessing':
                    _clear_after('preprocessing'); perform_preprocessing(); st.rerun()
                if section in ('automl', 'model_comparison', 'evaluation', 'feature_importance', 'imbalance'):
                    _clear_after('automl'); perform_automl(); st.rerun()
                add_message("assistant", "Dis-moi quelle partie je dois refaire : analyse, prétraitement, AutoML.")
                st.rerun()

            add_message("assistant", generate_response(intent=intent, context=context, llm_explainer=st.session_state.llm_explainer, message=prompt))
        except Exception as e:
            add_message("assistant", f"Désolé, une erreur est survenue: {e}")
        st.rerun()

if __name__ == "__main__":
    main()
