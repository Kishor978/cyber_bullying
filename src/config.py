import os

# Dataset Paths
VIDGEN_DATASET_PATH = r"E:\Cyberbullying\dataset\raw\Dynamically Generated Hate Dataset v0.2.2.csv"
DAVIDSON_DATASET_PATH = r"E:\Cyberbullying\dataset\raw\davidson.csv"
HATEMOJI_VALIDATION_PATH = r"E:\Cyberbullying\dataset\raw\HatemojiBuild\train.csv"

# Model and Output Paths
LOGISTIC_REGRESSION_MODEL_PATH = './models/logistic_regression_model.pkl'
BERT_MODEL_OUTPUT_DIR = './results/bert_model'
BILSTM_MODEL_OUTPUT_DIR = './results/bilstm_model'
EMOTION_FUSION_MODEL_OUTPUT_DIR = './results/emotion_fusion_model'

# GloVe Embedding Path (for BiLSTM)
GLOVE_PATH = r"E:\Cyberbullying\glove\glove.6B.100d.txt" # Adjust this path
LOGGING_DIR = r"E:\Cyberbullying\logs"  # Log directory for BERT Trainer

KAGGLE_WORKING_DIR = "/kaggle/working"
VIDGEN_DATASET_PATH = "/kaggle/input/cyberbullying/raw/Dynamically Generated Hate Dataset v0.2.2.csv"
DAVIDSON_DATASET_PATH = "/kaggle/input/cyberbullying/raw/davidson.csv"
HATEMOJI_VALIDATION_PATH = "/kaggle/input/cyberbullying/raw/HatemojiBuild/train.csv"
GLOVE_PATH= "/kaggle/input/cyberbullying/glove.6B.100d.txt"  # Adjust this path

LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(KAGGLE_WORKING_DIR, 'models', 'logistic_regression_model.pkl')
BERT_MODEL_OUTPUT_DIR = os.path.join(KAGGLE_WORKING_DIR, 'results', 'bert_model')
BILSTM_MODEL_OUTPUT_DIR = os.path.join(KAGGLE_WORKING_DIR, 'results', 'bilstm_model')
EMOTION_FUSION_MODEL_OUTPUT_DIR = os.path.join(KAGGLE_WORKING_DIR, 'results', 'emotion_fusion_model')

# Log directory for BERT Trainer
LOGGING_DIR = os.path.join(KAGGLE_WORKING_DIR, 'logs')

# General Settings
TFIDF_MAX_FEATURES = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2