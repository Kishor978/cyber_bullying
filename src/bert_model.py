# src/bert_model.py

import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Constants
TEST_SIZE = 0.2
RANDOM_STATE = 42
BERT_MODEL_OUTPUT_DIR = "./bert_output"

# Dataset
class CyberbullyingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# Tokenizer & Model
def get_bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def get_bert_model(num_labels=2):
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Metrics
def compute_bert_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='binary'),
        'recall': recall_score(labels, preds, average='binary'),
        'f1': f1_score(labels, preds, average='binary'),
    }

# Trainer
def create_bert_trainer(model, training_args_output_dir, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=training_args_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_bert_metrics
    )
    return trainer
