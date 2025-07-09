import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Assuming necessary imports from other parts of your project
from src.config import EMOTION_FUSION_MODEL_OUTPUT_DIR, LOGGING_DIR # Added for clarity, not directly used here

# Define the dataset class
class CyberbullyingFusionDataset(Dataset):
    def __init__(self, input_ids, attention_masks, emoji_scores, emotion_vectors, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.emoji_scores = emoji_scores
        self.emotion_vectors = emotion_vectors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Ensure labels are float for BCELoss
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'emoji_score': torch.tensor(self.emoji_scores[idx], dtype=torch.float),
            'emotion_vector': torch.tensor(self.emotion_vectors[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float) # Changed from torch.long to torch.float
        }

# Define the fusion model
class BERTEmojiEmotionClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', emotion_dim=768, dropout_prob=0.1): # Default emotion_dim for now
        super(BERTEmojiEmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_prob)

        # Assuming BERT's last hidden state is 768
        bert_output_dim = self.bert.config.hidden_size
        
        # Linear layer for emoji score
        self.emoji_fc = nn.Linear(1, 64) # Simple FC layer for emoji score
        
        # Linear layer for emotion vector (if it's not already 768 or compatible)
        # Adjust input dimension based on the actual size of your emotion vector
        self.emotion_fc = nn.Linear(emotion_dim, 128) # Map emotion_dim to a compatible size

        # Fusion layer: BERT output + Emoji FC output + Emotion FC output
        # Adjust the input dimension of the fusion_fc based on actual concatenated sizes
        # For simplicity, let's assume we map all to 256 for concatenation example
        # (bert_output_dim + 64 + 128) -> This is the input to fusion_fc
        self.fusion_fc = nn.Linear(bert_output_dim + 64 + 128, 256)
        
        self.classifier = nn.Linear(256, 1) # Output for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, emoji_score, emotion_vector):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :] # Use CLS token output
        
        emoji_features = self.emoji_fc(emoji_score.unsqueeze(1)) # Add a dimension for the single score
        
        # Ensure emotion_vector has the correct shape for the linear layer
        # If emotion_vector is already (batch_size, emotion_dim), no unsqueeze needed
        emotion_features = self.emotion_fc(emotion_vector)
        
        # Concatenate features
        # Ensure all features are 2D (batch_size, feature_dim) before concatenation
        combined_features = torch.cat((bert_output, emoji_features, emotion_features), dim=1)
        
        combined_features = self.dropout(combined_features)
        fusion_output = self.fusion_fc(combined_features)
        logits = self.classifier(fusion_output)
        return self.sigmoid(logits).squeeze(1) # Squeeze to make it (batch_size,)


# Training function
def train_fusion_model_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        emoji_score = batch['emoji_score'].to(device)
        emotion_vector = batch['emotion_vector'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, emoji_score, emotion_vector)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# Evaluation function
def evaluate_fusion_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emoji_score = batch['emoji_score'].to(device)
            emotion_vector = batch['emotion_vector'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, emoji_score, emotion_vector)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    
    # Ensure all_labels and all_predictions are lists or numpy arrays before conversion
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)

    # Convert to integer type for sklearn metrics if they were originally floats 0.0/1.0
    all_labels_int = all_labels_np.astype(int)
    all_predictions_int = all_predictions_np.astype(int)

    return avg_loss, all_predictions_int, all_labels_int