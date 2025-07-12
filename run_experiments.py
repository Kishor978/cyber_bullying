# run_experiments.py
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
import argparse # Import argparse
import numpy as np  
import joblib # For saving models with joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer
# Import all your modularized components
from src.config import (
    VIDGEN_DATASET_PATH, DAVIDSON_DATASET_PATH, HATEMOJI_VALIDATION_PATH,
    TFIDF_MAX_FEATURES, RANDOM_STATE, TEST_SIZE, GLOVE_PATH,
    BERT_MODEL_OUTPUT_DIR, BILSTM_MODEL_OUTPUT_DIR, EMOTION_FUSION_MODEL_OUTPUT_DIR,
    LOGISTIC_REGRESSION_MODEL_PATH # Added for LR model saving
)
from src.data_loader import load_vidgen_dataset, load_davidson_dataset, load_hatemoji_validation_dataset
from src.text_preprocessing import clean_text, vectorize_tfidf, clean_and_tokenize
from src.model_training import train_logistic_regression, save_model # save_model is for joblib models
from src.evaluation_metrics import print_classification_metrics, plot_confusion_matrix, plot_training_history
from src.bert_model import CyberbullyingDataset, get_bert_tokenizer, get_bert_model, create_bert_trainer, compute_bert_metrics
from src.bilstm_model import SimpleVocab, load_glove, TextDataset, bilstm_collate_fn, BiLSTMClassifier, train_bilstm_model, eval_bilstm_model, get_bilstm_predictions
from src.emotion_features import process_texts_for_emotion_features
from src.fusion_model import CyberbullyingFusionDataset, BERTEmojiEmotionClassifier, train_fusion_model_epoch, evaluate_fusion_model


# Ensure output directories exist
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/bert_model', exist_ok=True)
os.makedirs('./results/bilstm_model', exist_ok=True)    
os.makedirs('./results/emotion_fusion_model', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True) # For BERT trainer logs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_baseline_model():
    print("\n" + "="*80)
    print("                 Running Baseline (Logistic Regression) Model                 ")
    print("="*80 + "\n")

    # Vidgen Dataset
    vidgen_df = load_vidgen_dataset()
    vidgen_df['clean_text'] = vidgen_df['text'].apply(clean_text)
    X_vidgen, _ = vectorize_tfidf(vidgen_df['clean_text'], max_features=TFIDF_MAX_FEATURES)
    y_vidgen = vidgen_df['label'].values

    print("\n--- Training Logistic Regression on Vidgen Dataset ---")
    lr_model_vidgen, X_test_vidgen, y_test_vidgen = train_logistic_regression(X_vidgen, y_vidgen)
    y_pred_vidgen = lr_model_vidgen.predict(X_test_vidgen)
    print_classification_metrics(y_test_vidgen, y_pred_vidgen, "Vidgen Dataset (Logistic Regression)")
    plot_confusion_matrix(y_test_vidgen, y_pred_vidgen, "Confusion Matrix - Vidgen LR",
                          save_path="./results/baseline_vidgen_cm.png")
    save_model(lr_model_vidgen, LOGISTIC_REGRESSION_MODEL_PATH.replace(".pkl", "_vidgen.pkl")) # Save Vidgen LR model


    # Davidson Dataset
    davidson_df = load_davidson_dataset()
    davidson_df['clean_text'] = davidson_df['text'].apply(clean_text)
    X_davidson, _ = vectorize_tfidf(davidson_df['clean_text'], max_features=TFIDF_MAX_FEATURES)
    y_davidson = davidson_df['label'].values

    print("\n--- Training Logistic Regression on Davidson Dataset ---")
    lr_model_davidson, X_test_davidson, y_test_davidson = train_logistic_regression(X_davidson, y_davidson)
    y_pred_davidson = lr_model_davidson.predict(X_test_davidson)
    print_classification_metrics(y_test_davidson, y_pred_davidson, "Davidson Dataset (Logistic Regression)")
    plot_confusion_matrix(y_test_davidson, y_pred_davidson, "Confusion Matrix - Davidson LR",
                          save_path="./results/baseline_davidson_cm.png")
    save_model(lr_model_davidson, LOGISTIC_REGRESSION_MODEL_PATH.replace(".pkl", "_davidson.pkl")) # Save Davidson LR model


def run_bert_model_experiment():
    print("\n" + "="*80)
    print("                     Running BERT Model Experiment                    ")
    print("="*80 + "\n")

    # Load and split data
    df_bert = load_hatemoji_validation_dataset()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_bert['text'], df_bert['label'],
        test_size=TEST_SIZE, stratify=df_bert['label'], random_state=RANDOM_STATE
    )

    # Tokenize
    tokenizer = get_bert_tokenizer()
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Datasets
    train_dataset = CyberbullyingDataset(train_encodings, train_labels.tolist())
    test_dataset = CyberbullyingDataset(test_encodings, test_labels.tolist())

    # Model and Trainer
    model = get_bert_model(num_labels=2)
    trainer = create_bert_trainer(model, BERT_MODEL_OUTPUT_DIR, train_dataset, test_dataset)

    # Train
    print("\n--- Training BERT Model ---")
    trainer.train()

    # Save best model and tokenizer
    model.save_pretrained(BERT_MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(BERT_MODEL_OUTPUT_DIR)

    # Evaluate
    print("\n--- Evaluating BERT Model ---")
    eval_results = trainer.evaluate()
    print("BERT Evaluation Results:", eval_results)

    # Predict and visualize
    predictions = trainer.predict(test_dataset)
    y_true_bert = predictions.label_ids
    y_pred_bert = predictions.predictions.argmax(axis=1)

    print_classification_metrics(y_true_bert, y_pred_bert, "BERT Model")
    plot_confusion_matrix(
        y_true_bert, y_pred_bert, 
        "BERT Confusion Matrix", 
        save_path="./results/bert_confusion_matrix.png"
    )

def run_bilstm_model_experiment():
    print("\n" + "="*80)
    print("                     Running BiLSTM Model Experiment                    ")
    print("="*80 + "\n")

    df_bilstm = load_vidgen_dataset() # BiLSTM notebook used Dynamically Generated Hate Dataset
    df_bilstm['tokens'] = df_bilstm['text'].apply(clean_and_tokenize) #

    vocab = SimpleVocab(df_bilstm['tokens'].tolist(), min_freq=2) #
    PAD_IDX = vocab['<pad>'] #

    embedding_matrix = load_glove(GLOVE_PATH, vocab, dim=100) #

    train_texts, test_texts, train_labels, test_labels = train_test_split( #
        df_bilstm['tokens'], df_bilstm['label'], test_size=TEST_SIZE, stratify=df_bilstm['label'], random_state=RANDOM_STATE) #

    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), vocab) #
    test_dataset = TextDataset(test_texts.tolist(), test_labels.tolist(), vocab) #

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              collate_fn=lambda b: bilstm_collate_fn(b, PAD_IDX)) #
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             collate_fn=lambda b: bilstm_collate_fn(b, PAD_IDX)) #

    model = BiLSTMClassifier(embedding_matrix, hidden_dim=128, output_dim=1).to(device) #
    criterion = torch.nn.BCELoss() #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #

    train_losses, train_accuracies = [], [] #
    val_losses, val_accuracies = [], [] #

    print("\n--- Training BiLSTM Model ---")
    for epoch in range(1, 6): #
        train_loss, train_acc = train_bilstm_model(model, train_loader, optimizer, criterion, device) #
        val_loss, val_acc = eval_bilstm_model(model, test_loader, criterion, device) #

        train_losses.append(train_loss) #
        train_accuracies.append(train_acc) #
        val_losses.append(val_loss) #
        val_accuracies.append(val_acc) #

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}") #

    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, "BiLSTM",
                          save_path="./results/bilstm_training_history.png")

    print("\n--- Evaluating BiLSTM Model ---")
    y_pred_bilstm, y_true_bilstm = get_bilstm_predictions(model, test_loader, device)
    print_classification_metrics(y_true_bilstm, y_pred_bilstm, "BiLSTM Model")
    plot_confusion_matrix(y_true_bilstm, y_pred_bilstm, "BiLSTM Confusion Matrix",
                          save_path="./results/bilstm_confusion_matrix.png")

    # Save the BiLSTM model
    torch.save(model.state_dict(), BILSTM_MODEL_OUTPUT_DIR + "/bilstm_model.pth")
    print(f"BiLSTM model saved to {BILSTM_MODEL_OUTPUT_DIR}/bilstm_model.pth")


def run_emotion_fusion_model_experiment():
    print("\n" + "="*80)
    print("                Running Emotion Fusion Model Experiment               ")
    print("="*80 + "\n")

    df_fusion = load_hatemoji_validation_dataset()
    print(f"🔹 Loaded dataset: {df_fusion.shape[0]} rows")

    df_fusion = process_texts_for_emotion_features(df_fusion)
    print(f"🔹 After emotion feature extraction: {df_fusion.shape}")

    # Debug: check issues with emotion vectors
    zero_vectors = df_fusion['emotion_vector'].apply(lambda x: isinstance(x, list) and np.all(np.array(x) == 0.0)).sum()
    invalid_lengths = df_fusion['emotion_vector'].apply(lambda x: not isinstance(x, list) or len(x) != 7).sum()
    print(f"⚠️  Rows with all-zero emotion vectors: {zero_vectors}")
    print(f"❌ Rows with invalid-length vectors: {invalid_lengths}")
    print(f"✅ Rows with valid non-zero vectors: {df_fusion.shape[0] - zero_vectors - invalid_lengths}")

    # Filter only valid rows
    df_fusion = df_fusion[df_fusion['text'].notnull() & df_fusion['text'].str.strip().astype(bool)]
    df_fusion = df_fusion[df_fusion['emoji_score'].notnull()]
    df_fusion = df_fusion[df_fusion['emotion_vector'].apply(lambda x: isinstance(x, list) and len(x) == 7)]
    print(f"🧹 After cleaning: {df_fusion.shape[0]} rows remaining")

    if df_fusion.empty:
        print("❌ ERROR: All rows dropped during cleaning. Check emotion model or data quality.")
        return

    # Optional: Remove zero vectors if needed
    # df_fusion = df_fusion[df_fusion['emotion_vector'].apply(lambda x: sum(x) > 0.0)]

    # Tokenizer sanity check
    print("\n✅ Tokenizer input check:")
    print(f"Number of texts: {len(df_fusion['text'])}")
    print(f"Example text: {df_fusion['text'].iloc[0]}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encodings = tokenizer(list(df_fusion['text']), truncation=True, padding=True, max_length=128)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    train_data, test_data = train_test_split(
        df_fusion, test_size=TEST_SIZE, stratify=df_fusion['label'], random_state=RANDOM_STATE)

    # Create Dataset and DataLoader instances
    train_dataset = CyberbullyingFusionDataset(
        input_ids=tokenizer(list(train_data['text']), truncation=True, padding=True, max_length=128)['input_ids'],
        attention_masks=tokenizer(list(train_data['text']), truncation=True, padding=True, max_length=128)['attention_mask'],
        emoji_scores=list(train_data['emoji_score']),
        emotion_vectors=list(train_data['emotion_vector']),
        labels=list(train_data['label'])
    )

    test_dataset = CyberbullyingFusionDataset(
        input_ids=tokenizer(list(test_data['text']), truncation=True, padding=True, max_length=128)['input_ids'],
        attention_masks=tokenizer(list(test_data['text']), truncation=True, padding=True, max_length=128)['attention_mask'],
        emoji_scores=list(test_data['emoji_score']),
        emotion_vectors=list(test_data['emotion_vector']),
        labels=list(test_data['label'])
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = BERTEmojiEmotionClassifier(emotion_dim=7).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print("\n--- Training Emotion Fusion Model ---")
    for epoch in range(1, 6):
        train_loss, train_acc = train_fusion_model_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_pred, y_true = evaluate_fusion_model(model, test_loader, criterion, device)
        acc_val = accuracy_score(y_true, y_pred)
        prec_val = precision_score(y_true, y_pred)
        rec_val = recall_score(y_true, y_pred)
        f1_val = f1_score(y_true, y_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(acc_val)

        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Accuracy: {acc_val:.4f}, Precision: {prec_val:.4f}, Recall: {rec_val:.4f}, F1: {f1_val:.4f}")

    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies,
                          "Emotion Fusion Model", save_path="./results/emotion_fusion_training_history.png")

    print("\n--- Evaluating Emotion Fusion Model ---")
    _, y_pred_fusion, y_true_fusion = evaluate_fusion_model(model, test_loader, criterion, device)
    print_classification_metrics(y_true_fusion, y_pred_fusion, "Emotion Fusion Model")
    plot_confusion_matrix(y_true_fusion, y_pred_fusion, "Confusion Matrix - Proposed Model",
                          save_path="./results/emotion_fusion_confusion_matrix.png")

    # Save the trained model
    torch.save(model.state_dict(), EMOTION_FUSION_MODEL_OUTPUT_DIR + "/emotion_fusion_model.pth")
    print(f"✅ Emotion Fusion model saved to {EMOTION_FUSION_MODEL_OUTPUT_DIR}/emotion_fusion_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cyberbullying Detection Experiments.")
    parser.add_argument('--all', action='store_true', help='Run all experiments (default if no specific experiment is chosen).')
    parser.add_argument('--baseline', action='store_true', help='Run the Baseline (Logistic Regression) model experiment.')
    parser.add_argument('--bert', action='store_true', help='Run the BERT model experiment.')
    parser.add_argument('--bilstm', action='store_true', help='Run the BiLSTM model experiment.')
    parser.add_argument('--emotion', action='store_true', help='Run the Emotion Fusion model experiment.')
    args = parser.parse_args()

    # Ensure a consistent random seed for all experiments
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # Determine which experiments to run
    run_all = not (args.baseline or args.bert or args.bilstm or args.emotion) or args.all

    if run_all or args.baseline:
        run_baseline_model()
    if run_all or args.bert:
        run_bert_model_experiment()
    if run_all or args.bilstm:
        run_bilstm_model_experiment()
    if run_all or args.emotion:
        run_emotion_fusion_model_experiment()

    print("\nAll selected experiments finished!")