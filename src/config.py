import os

# General Settings
TFIDF_MAX_FEATURES = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Dataset Paths
VIDGEN_DATASET_PATH = r"E:\Cyberbullying\dataset\raw\Dynamically Generated Hate Dataset v0.2.2.csv"
DAVIDSON_DATASET_PATH = r"E:\Cyberbullying\dataset\raw\davidson.csv"
COMBINED = r"E:\Cyberbullying\dataset\11\combined_dataset.csv"

# Model and Output Paths
LOGISTIC_REGRESSION_MODEL_PATH = r'E:\Cyberbullying\models\logistic_regression_model.pkl'
BERT_MODEL_OUTPUT_DIR = r'E:\Cyberbullying\results\bert_model'
BILSTM_MODEL_OUTPUT_DIR = r'E:\Cyberbullying\results\bilstm_model'
EMOTION_FUSION_MODEL_OUTPUT_DIR = r'E:\Cyberbullying\results\emotion_fusion_model'

# GloVe Embedding Path (for BiLSTM)
GLOVE_PATH = r"E:\Cyberbullying\glove\glove.6B.100d.txt"
LOGGING_DIR = r"E:\Cyberbullying\logs"  # Log directory for BERT Trainer

# Deployment paths
DEPLOYED_MODELS_DIR = r"E:\Cyberbullying\deployed_models"
DEPLOYED_LOGISTIC_REGRESSION_MODEL= os.path.join(DEPLOYED_MODELS_DIR, 'logistic_regression_model')
DEPLOYED_BERT_MODEL_DIR = os.path.join(DEPLOYED_MODELS_DIR, 'bert_model')
DEPLOYED_BILSTM_MODEL_DIR = os.path.join(DEPLOYED_MODELS_DIR, 'bilstm_model')
DEPLOYED_EMOTION_FUSION_MODEL_DIR = os.path.join(DEPLOYED_MODELS_DIR, 'emotion_fusion_model')

# Streamlit deployment configuration
STREAMLIT_PORT = 8502
STREAMLIT_THEME = "light"

# Kaggle Paths (commented out)
"""
KAGGLE_WORKING_DIR = "/kaggle/working"
VIDGEN_DATASET_PATH = "/kaggle/input/cyberbullying/raw/Dynamically Generated Hate Dataset v0.2.2.csv"
DAVIDSON_DATASET_PATH = "/kaggle/input/cyberbullying/raw/davidson.csv"
HATEMOJI_VALIDATION_PATH = "/kaggle/input/cyberbullying/raw/HatemojiBuild/train.csv"
GLOVE_PATH= "/kaggle/input/cyberbullying/glove.6B.100d.txt"
OMG_PATH='/kaggle/input/cyberbullying/OMG.csv'
COMBINED = "/kaggle/input/cyberbullying/11/combined_dataset.csv"
LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(KAGGLE_WORKING_DIR, 'models', 'logistic_regression_model.pkl')
BERT_MODEL_OUTPUT_DIR = os.path.join(KAGGLE_WORKING_DIR, 'results', 'bert_model')
BILSTM_MODEL_OUTPUT_DIR = os.path.join(KAGGLE_WORKING_DIR, 'results', 'bilstm_model')
EMOTION_FUSION_MODEL_OUTPUT_DIR = os.path.join(KAGGLE_WORKING_DIR, 'results', 'emotion_fusion_model')
LOGGING_DIR = os.path.join(KAGGLE_WORKING_DIR, 'logs')
"""