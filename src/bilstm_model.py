# src/bilstm_model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np

class SimpleVocab: #
    def __init__(self, token_lists, min_freq=2): #
        self.freqs = Counter() #
        for tokens in token_lists: #
            self.freqs.update(tokens) #

        self.itos = ['<pad>', '<unk>'] + [tok for tok, freq in self.freqs.items() if freq >= min_freq] #
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)} #

    def __getitem__(self, token): #
        return self.stoi.get(token, self.stoi['<unk>']) #

    def __len__(self): #
        return len(self.itos) #

def load_glove(path, vocab, dim=100): #
    glove = {} #
    with open(path, encoding='utf8') as f: #
        for line in f: #
            tokens = line.split() #
            word = tokens[0] #
            vec = np.array(tokens[1:], dtype=np.float32) #
            glove[word] = vec #

    embedding_matrix = np.zeros((len(vocab), dim), dtype=np.float32) #
    for word, idx in vocab.stoi.items(): #
        if word in glove: #
            embedding_matrix[idx] = glove[word] #
        else: #
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(dim,)).astype(np.float32) #

    return torch.tensor(embedding_matrix, dtype=torch.float32) #

class TextDataset(torch.utils.data.Dataset): #
    def __init__(self, texts, labels, vocab): #
        self.texts = texts #
        self.labels = labels #
        self.vocab = vocab # New: pass vocab
        self.PAD_IDX = vocab['<pad>'] # New: store PAD_IDX

    def __len__(self): #
        return len(self.labels) #

    def __getitem__(self, idx): #
        tokens = self.texts[idx] #
        indices = [self.vocab[token] for token in tokens] #
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32) #

def bilstm_collate_fn(batch, PAD_IDX): #
    texts, labels = zip(*batch) #
    lengths = [len(x) for x in texts] #
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX) #

    return ( #
        padded_texts,  # LongTensor #
        torch.stack(labels),  # Proper stacking without changing dtype #
        torch.tensor(lengths, dtype=torch.long) #
    )

class BiLSTMClassifier(nn.Module): #
    def __init__(self, embedding_matrix, hidden_dim, output_dim, dropout=0.5): #
        super().__init__() #
        vocab_size, embedding_dim = embedding_matrix.shape #
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False) #
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True) #
        self.dropout = nn.Dropout(dropout) #
        self.fc = nn.Linear(hidden_dim * 2, output_dim) #
        self.sigmoid = nn.Sigmoid() #

    def forward(self, text, lengths): #
        embedded = self.embedding(text) #
        packed_output, (hidden, _) = self.lstm(embedded) #
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1) #
        out = self.dropout(hidden_cat) #
        return self.sigmoid(self.fc(out)).squeeze() #

def train_bilstm_model(model, loader, optimizer, criterion, device): #
    model.train() #
    total_loss, correct, total = 0, 0, 0 #
    for texts, labels, lengths in loader: #
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device) #
        optimizer.zero_grad() #
        outputs = model(texts, lengths) #
        loss = criterion(outputs, labels) #
        loss.backward() #
        optimizer.step() #

        total_loss += loss.item() #
        preds = (outputs >= 0.5).long() #
        correct += (preds == labels).sum().item() #
        total += labels.size(0) #

    acc = correct / total #
    return total_loss / len(loader), acc #

def eval_bilstm_model(model, loader, criterion, device): #
    model.eval() #
    total_loss, correct, total = 0, 0, 0 #
    with torch.no_grad(): #
        for texts, labels, lengths in loader: #
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device) #
            outputs = model(texts, lengths) #
            loss = criterion(outputs, labels) #

            total_loss += loss.item() #
            preds = (outputs >= 0.5).long() #
            correct += (preds == labels).sum().item() #
            total += labels.size(0) #

    acc = correct / total #
    return total_loss / len(loader), acc #

def get_bilstm_predictions(model, loader, device): # Modified from bilstm.ipynb evaluate_model function
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            outputs = model(texts, lengths)
            predicted = (outputs >= 0.5).long()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)