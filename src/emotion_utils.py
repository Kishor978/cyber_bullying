from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Emotion model
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_model.eval().to(device)

@torch.no_grad()
def get_emotion_vector(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    logits = emotion_model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    return probs.squeeze(0).detach().cpu()  # shape: [6]


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        emotion_vec = get_emotion_vector(text)  # [6]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'emotion_vec': emotion_vec,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_f1 = 0
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, f1, model):
        if f1 > self.best_f1 + self.delta:
            self.best_f1 = f1
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    loop = tqdm(dataloader, desc="Training", leave=False)

    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        emotion_vec = batch['emotion_vec'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, emotion_vec=emotion_vec)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    acc = total_correct / len(dataloader.dataset)
    return total_loss / len(dataloader), acc
    

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [], []

    loop = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_vec = batch['emotion_vec'].to(device)  # ✅ ADD THIS
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, emotion_vec=emotion_vec)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total_correct += (preds == labels).sum().item()

            loop.set_postfix(loss=loss.item())

    acc = total_correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


def final_evaluation(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ensure all_labels and all_predictions are lists or numpy arrays before conversion
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_preds)

    # Convert to integer type for sklearn metrics if they were originally floats 0.0/1.0
    all_labels_int = all_labels_np.astype(int)
    all_predictions_int = all_predictions_np.astype(int)
    return all_predictions_int, all_labels_int