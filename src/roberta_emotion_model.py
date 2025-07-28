import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaCNNWithEmotion(nn.Module):
    def __init__(self, roberta_model='roberta-base', num_classes=2, dropout=0.5):
        super(RobertaCNNWithEmotion, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(100 + 7, num_classes)  # +6 for emotion vector

    def forward(self, input_ids, attention_mask, emotion_vec):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (B, L, 768)
        x = x.permute(0, 2, 1)         # (B, 768, L)
        x = self.conv1(x)              # (B, 100, L)
        x = self.relu(x)
        x = torch.max(x, dim=2).values # (B, 100)
        x = self.dropout(x)
        x = torch.cat([x, emotion_vec], dim=1)  # Concatenate emotion vector
        logits = self.fc(x)
        return logits
