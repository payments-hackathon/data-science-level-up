import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_prob is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['auc'] = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics['pr_auc'] = auc(recall, precision)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(8, 6)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()

def plot_roc_curve(y_true, y_prob, figsize=(8, 6)):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, figsize=(8, 6)):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('plots/precision_recall_curve.png')
    plt.show()

def plot_training_history(train_losses, val_losses=None, train_accs=None, val_accs=None, figsize=(12, 4)):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss')
    if val_losses:
        axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    
    # F1/Accuracy plot
    if train_accs:
        axes[1].plot(train_accs, label='Train F1' if len(train_accs) <= 10 else 'Train Acc')
    if val_accs:
        axes[1].plot(val_accs, label='Val F1' if len(val_accs) <= 10 else 'Val Acc')
    axes[1].set_title('F1 Score' if val_accs and len(val_accs) <= 10 else 'Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.show()

def evaluate_model(model, dataloader, device='cpu', criterion=None):
    """Evaluate PyTorch model"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader) if criterion else None
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), avg_loss

def print_classification_report(y_true, y_pred, target_names=None):
    """Print detailed classification report"""
    print(classification_report(y_true, y_pred, target_names=target_names))

def model_summary(model, input_size):
    """Print model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input size: {input_size}")
    print(f"Model: {model.__class__.__name__}")