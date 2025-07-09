# src/model_training.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE
import joblib # For saving/loading models

def train_logistic_regression(X, y):
    """Trains a Logistic Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y) #
    model = LogisticRegression(max_iter=1000) #
    model.fit(X_train, y_train) #
    return model, X_test, y_test

def save_model(model, path):
    """Saves a trained model to a file."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """Loads a trained model from a file."""
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model