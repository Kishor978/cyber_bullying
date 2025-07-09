# src/fusion_model.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class CyberbullyingFusionDataset(torch.utils.data.Dataset): #
    def __init__(self, input_ids, attention_masks, emoji_scores, emotion_vectors, labels): #
        self.input_ids = input_ids #
        self.attention_masks = attention_masks #
        self.emoji_scores = emoji_scores #
        self.emotion_vectors = emotion_vectors #
        self.labels = labels #

    def __len__(self): #
        return len(self.labels) #

    def __getitem__(self, idx): #
        return { #
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long), #
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long), #
            'emoji_score': torch.tensor(self.emoji_scores[idx], dtype=torch.float32), #
            'emotion_vector': torch.tensor(self.emotion_vectors[idx], dtype=torch.float32), #
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32) #
        }

class BERTEmojiEmotionClassifier(nn.Module): #
    def __init__(self, emotion_dim=6, hidden_dim=256, dropout=0.3): #
        super().__init__() #
        self.bert = BertModel.from_pretrained('bert-base-uncased') #

        # Optional: project emotion and emoji features if needed #
        self.emoji_proj = nn.Linear(1, 8) #
        self.emotion_proj = nn.Linear(emotion_dim, 32) #

        # Fusion + classifier #
        self.classifier = nn.Sequential( #
            nn.Linear(768 + 8 + 32, hidden_dim), #
            nn.ReLU(), #
            nn.Dropout(dropout), #
            nn.Linear(hidden_dim, 1), #
            nn.Sigmoid() #
        )

    def forward(self, input_ids, attention_mask, emoji_score, emotion_vector): #
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) #
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token #

        emoji_feat = self.emoji_proj(emoji_score.unsqueeze(1))  # shape: [B, 8] #
        emotion_feat = self.emotion_proj(emotion_vector)        # shape: [B, 32] #

        combined = torch.cat((cls_embedding, emoji_feat, emotion_feat), dim=1) #
        out = self.classifier(combined) #
        return out.squeeze() #

def train_fusion_model_epoch(model, loader, optimizer, criterion, device): # Modified from emotion.ipynb
    model.train() #
    total_loss, correct = 0, 0 #

    for batch in loader: #
        input_ids = batch['input_ids'].to(device) #
        attention_mask = batch['attention_mask'].to(device) #
        emoji_score = batch['emoji_score'].to(device) #
        emotion_vector = batch['emotion_vector'].to(device) #
        labels = batch['labels'].to(device) #

        optimizer.zero_grad() #
        outputs = model(input_ids, attention_mask, emoji_score, emotion_vector) #
        loss = criterion(outputs, labels) #
        loss.backward() #
        optimizer.step() #

        total_loss += loss.item() #
        preds = (outputs >= 0.5).float() #
        correct += (preds == labels).sum().item() #

    acc = correct / len(loader.dataset) #
    return total_loss / len(loader), acc #

def evaluate_fusion_model(model, loader, criterion, device): # Modified from emotion.ipynb
    model.eval() #
    all_preds, all_labels = [], [] #
    total_loss = 0 #

    with torch.no_grad(): #
        for batch in loader: #
            input_ids = batch['input_ids'].to(device) #
            attention_mask = batch['attention_mask'].to(device) #
            emoji_score = batch['emoji_score'].to(device) #
            emotion_vector = batch['emotion_vector'].to(device) #
            labels = batch['labels'].to(device) #

            outputs = model(input_ids, attention_mask, emoji_score, emotion_vector) #
            loss = criterion(outputs, labels) #
            total_loss += loss.item() #

            preds = (outputs >= 0.5).long().cpu().numpy() #
            all_preds.extend(preds) #
            all_labels.extend(labels.cpu().numpy()) #

    # Metrics calculation moved to evaluation_metrics.py
    return total_loss / len(loader), all_preds, all_labels