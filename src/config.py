# src/config.py

# Dataset Paths
# VIDGEN_DATASET_PATH = r"E:\Cyberbullying\dataset\raw\Dynamically Generated Hate Dataset v0.2.2.csv"
# DAVIDSON_DATASET_PATH = r"E:\Cyberbullying\dataset\raw\davidson.csv"
# HATEMOJI_VALIDATION_PATH = r"E:\Cyberbullying\dataset\raw\HatemojiBuild\validation.csv"


VIDGEN_DATASET_PATH = "/kaggle/input/cyberbullying/raw/Dynamically Generated Hate Dataset v0.2.2.csv"
DAVIDSON_DATASET_PATH = "/kaggle/input/cyberbullying/raw/davidson-dataset/davidson.csv"
HATEMOJI_VALIDATION_PATH = "/kaggle/input/cyberbullying/raw/hatemojibuild/validation.csv"

# Model and Output Paths
LOGISTIC_REGRESSION_MODEL_PATH = './models/logistic_regression_model.pkl'
BERT_MODEL_OUTPUT_DIR = './results/bert_model'
BILSTM_MODEL_OUTPUT_DIR = './results/bilstm_model'
EMOTION_FUSION_MODEL_OUTPUT_DIR = './results/emotion_fusion_model'

# GloVe Embedding Path (for BiLSTM)
# GLOVE_PATH = r"E:\Cyberbullying\glove\glove.6B.100d.txt" # Adjust this path
GLOVE_PATH= "/kaggle/input/cyberbullying/glove.6B.100d.txt"  # Adjust this path
# General Settings
TFIDF_MAX_FEATURES = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2