import os
import sys
# Add project root to path so we can import modules - MUST BE BEFORE OTHER IMPORTS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pickle
import joblib
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer
import torch.nn.functional as F

from src.text_preprocessing import clean_text
from src.bilstm_model import BiLSTMClassifier
from src.roberta_emotion_model import RobertaCNNWithEmotion
from src.emotion_utils import get_emotion_vector

from src.config import (
    DEPLOYED_BERT_MODEL_DIR,
    DEPLOYED_BILSTM_MODEL_DIR,
    DEPLOYED_EMOTION_FUSION_MODEL_DIR,
    DEPLOYED_LOGISTIC_REGRESSION_MODEL,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelPredictor:
    """Class to handle model loading and prediction for multiple models."""
    
    def __init__(self, model_type='emotion_fusion'):
        """
        Initialize model predictor.
        
        Args:
            model_type (str): Type of model to use ('bert', 'bilstm', 'emotion_fusion', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.vocab = None
        self.vectorizer = None
        
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
        
        # Return prediction and confidence
        result = {
            'prediction': prediction,
            'prediction_text': 'Bullying' if prediction == 1 else 'Non-bullying',
            'confidence': confidence,
            'model_type': self.model_type
        }
        
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
