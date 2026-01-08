"""
Configuration globale du projet ChatAutoML-Bot
"""

# Nom du chatbot (personnalisable)
CHATBOT_NAME = "TabularAI"  # Nom du chatbot
CHATBOT_DESCRIPTION = "Assistant intelligent pour l'AutoML sur données tabulaires"
CHATBOT_ICON = "🚀"  # Emoji ou icône pour le chatbot

# Limites de fichiers
MAX_FILE_SIZE_MB = 100

# Paramètres de preprocessing
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Paramètres de validation croisée
CV_FOLDS = 5

# Paramètres de recherche d'hyperparamètres
SEARCH_METHOD = 'grid'  # 'grid' ou 'random'
N_ITER_RANDOM_SEARCH = 20

# Stratégie de rééchantillonnage
RESAMPLING_STRATEGY = 'auto'  # 'auto', 'smote', 'oversample', 'undersample', 'smote_tomek', None

# Configuration LLM
LLM_PROVIDER = 'ollama'  # 'ollama', 'openai', 'mistral', 'huggingface', 'template'
OLLAMA_MODEL = 'llama2'  # Modèle Ollama à utiliser (ex: 'llama2', 'mistral', 'codellama', 'phi')
OLLAMA_BASE_URL = 'http://localhost:11434'  # URL de base pour Ollama

# Pour OpenAI/Mistral (si utilisé)
# LLM_API_KEY = "votre_cle_api_ici"  # Décommentez et ajoutez votre clé

# Métriques par défaut
DEFAULT_CLASSIFICATION_METRIC = 'accuracy'
DEFAULT_REGRESSION_METRIC = 'r2'

