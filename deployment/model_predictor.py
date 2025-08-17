import os
import sys
import time
import uuid
import random
import torch
import pickle
import joblib
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer
import torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_preprocessing import clean_text
from src.bilstm_model import BiLSTMClassifier
from src.roberta_emotion_model import RobertaCNNWithEmotion
from src.emotion_utils import get_emotion_vector

# Setup logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_predictor")

# Import for metrics logging (optional - will be disabled if not available)
try:
    from monitoring.metrics_logger import MetricsLogger
    METRICS_AVAILABLE = True
    logger.info("Successfully imported MetricsLogger")
except ImportError as e:
    METRICS_AVAILABLE = False
    logger.error(f"Could not import MetricsLogger: {str(e)}")
    logger.error(f"Python path: {sys.path}")

from src.config import (
    DEPLOYED_BERT_MODEL_DIR,
    DEPLOYED_BILSTM_MODEL_DIR,
    DEPLOYED_EMOTION_FUSION_MODEL_DIR,
    DEPLOYED_LOGISTIC_REGRESSION_MODEL,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelPredictor:
    """Class to handle model loading and prediction for multiple models."""
    
    def __init__(self, model_type='emotion_fusion', enable_metrics=True):
        """
        Initialize model predictor.
        
        Args:
            model_type (str): Type of model to use ('bert', 'bilstm', 'emotion_fusion', 'logistic')
            enable_metrics (bool): Whether to enable metrics logging to Grafana/InfluxDB
        """
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.vocab = None
        self.vectorizer = None
        
        # Initialize metrics logger if available
        self.metrics_logger = None
        if METRICS_AVAILABLE and enable_metrics:
            try:
                logger.info("Initializing MetricsLogger...")
                self.metrics_logger = MetricsLogger(
                    enabled=True,
                    # Default to environment variable or use the token from docker-compose
                    token=os.getenv("INFLUXDB_TOKEN", "cyberbullying_token"),
                    url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
                    debug=True
                )
                logger.info("MetricsLogger initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize metrics logger: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Load the appropriate model
        self.load_model()
    
    def load_model(self):
        """Load the selected model and required components."""
        if self.model_type == 'bert':
            # Load BERT model and tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(DEPLOYED_BERT_MODEL_DIR)
            self.model = BertForSequenceClassification.from_pretrained(DEPLOYED_BERT_MODEL_DIR)
            self.model.to(device)
            self.model.eval()
            
        elif self.model_type == 'bilstm':
            # Load BiLSTM model and vocabulary
            vocab_path = os.path.join(DEPLOYED_BILSTM_MODEL_DIR, 'bilstm_vocab.pkl')
            model_path = os.path.join(DEPLOYED_BILSTM_MODEL_DIR, 'bilstm_model.pth')
            
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            # We need to know embedding dimensions - assuming 100D GloVe
            embedding_matrix = torch.zeros((len(self.vocab), 100))
            self.model = BiLSTMClassifier(embedding_matrix, hidden_dim=128, output_dim=1)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            
        elif self.model_type == 'emotion_fusion':
            # Load Emotion Fusion model
            model_path = os.path.join(DEPLOYED_EMOTION_FUSION_MODEL_DIR, 'emotion_fusion_model.pth')
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaCNNWithEmotion()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            
        elif self.model_type == 'logistic':
            # Load Logistic Regression model and vectorizer
            model_path =os.path.join(DEPLOYED_LOGISTIC_REGRESSION_MODEL,"logistic_regression_model_davidson.pkl")
            vectorizer_path = os.path.join(DEPLOYED_LOGISTIC_REGRESSION_MODEL,"logistic_regression_model_tfidf_davidson.pkl")

            with open(model_path, 'rb') as f:
                self.model = joblib.load(f)
                
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = joblib.load(f)
    
    def predict(self, text):
        """
        Predict whether text is cyberbullying or not.
        
        Args:
            text (str): Input text for prediction
            
        Returns:
            dict: Dictionary containing prediction results and confidence
        """
        # Start timer for latency measurement
        start_time = time.time()
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        if self.model_type == 'bert':
            # Tokenize input
            inputs = self.tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
            
        elif self.model_type == 'bilstm':
            # Tokenize and convert to indices
            from src.text_preprocessing import clean_and_tokenize
            tokens = clean_and_tokenize(text)
            indices = [self.vocab[token] for token in tokens]
            
            # Convert to tensor
            text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
            length_tensor = torch.tensor([len(indices)], dtype=torch.long).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(text_tensor, length_tensor)
                prob = torch.sigmoid(output).item()
                prediction = 1 if prob >= 0.5 else 0
                confidence = prob if prediction == 1 else 1 - prob
            
        elif self.model_type == 'emotion_fusion':
            # Tokenize input
            encoding = self.tokenizer(
                cleaned_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # Get emotion vector
            emotion_vec = get_emotion_vector(cleaned_text)
            
            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            emotion_vec = emotion_vec.to(device).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, emotion_vec=emotion_vec)
                probs = F.softmax(outputs, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
            
        elif self.model_type == 'logistic':
            # Transform text using TF-IDF
            text_vector = self.vectorizer.transform([cleaned_text]).toarray()
            
            # Get prediction
            prediction = self.model.predict(text_vector)[0]
            confidence = self.model.predict_proba(text_vector)[0][prediction]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Return prediction and confidence
        result = {
            'prediction': prediction,
            'prediction_text': 'Bullying' if prediction == 1 else 'Non-bullying',
            'confidence': confidence,
            'model_type': self.model_type,
            'latency_ms': latency_ms
        }
        
        # Log metrics if available
        if self.metrics_logger:
            try:
                # Generate a unique ID for this prediction
                text_id = str(uuid.uuid4())[:8]
                
                # Log the prediction metrics
                self.metrics_logger.log_prediction(
                    model_name=self.model_type,
                    text_id=text_id,
                    prediction=prediction,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    text_length=len(text)
                )
                
                # Log system metrics occasionally to avoid overloading the database
                if random.random() < 0.1:  # Log approximately every 10th prediction
                    self.metrics_logger.log_system_metrics()
            except Exception as e:
                # Don't let metrics logging failure affect the model prediction
                print(f"Warning: Failed to log metrics: {str(e)}")
        
        return result


# Example usage
if __name__ == "__main__":
    # Test prediction
    predictor = ModelPredictor(model_type='emotion_fusion')
    text = "Hey black, you are such a stupid person, I hate you!"
    result = predictor.predict(text)
    print(f"Text: {text}")
    print(f"Prediction: {result['prediction_text']}")
    print(f"Confidence: {result['confidence']:.4f}")
