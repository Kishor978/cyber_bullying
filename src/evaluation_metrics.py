# src/evaluation_metrics.py
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def print_classification_metrics(y_true, y_pred, dataset_name="Dataset"):
    """Prints accuracy, precision, recall, F1-score, and the full classification report."""
    accuracy = accuracy_score(y_true, y_pred) #
    precision = precision_score(y_true, y_pred) #
    recall = recall_score(y_true, y_pred) #
    f1 = f1_score(y_true, y_pred) #

    print(f"\n--- Metrics for {dataset_name} ---")
    print(f"Accuracy:  {accuracy:.4f}") #
    print(f"Precision: {precision:.4f}") #
    print(f"Recall:    {recall:.4f}") #
    print(f"F1-Score:  {f1:.4f}") #
    print(f"\nClassification Report of {dataset_name}:\n", classification_report(y_true, y_pred)) #

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plots and optionally saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred) #
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', #
                xticklabels=['Non-bullying', 'Bullying'], #
                yticklabels=['Non-bullying', 'Bullying']) #
    plt.xlabel('Predicted') #
    plt.ylabel('Actual') #
    plt.title(title) #
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.show() #
    plt.close()

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, title_prefix="", save_path=None): # From bilstm.ipynb
    """Plots training and validation loss/accuracy over epochs."""
    epochs = range(1, len(train_losses) + 1) #

    plt.figure(figsize=(12, 5)) #

    # Loss plot
    plt.subplot(1, 2, 1) #
    plt.plot(epochs, train_losses, label='Train Loss') #
    plt.plot(epochs, val_losses, label='Validation Loss') #
    plt.title(f'{title_prefix} Loss per Epoch') #
    plt.xlabel('Epoch') #
    plt.ylabel('Loss') #
    plt.legend() #

    # Accuracy plot
    plt.subplot(1, 2, 2) #
    plt.plot(epochs, train_accuracies, label='Train Accuracy') #
    plt.plot(epochs, val_accuracies, label='Validation Accuracy') #
    plt.title(f'{title_prefix} Accuracy per Epoch') #
    plt.xlabel('Epoch') #
    plt.ylabel('Accuracy') #
    plt.legend() #

    plt.tight_layout() #
    if save_path:
        plt.savefig(save_path)
        print(f"Saved training history plot to {save_path}")
    plt.show() #
    plt.close()