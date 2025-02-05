import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import os
plt.style.use('seaborn-v0_8')

def plot_curves(train_losses, val_losses, val_accs, mmd_losses=None, save_dir="results/figures"):
    """绘制训练曲线：损失曲线、准确率曲线、MMD损失曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(val_accs, label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # MMD损失曲线（如果存在）
    if mmd_losses:
        plt.subplot(1, 3, 3)
        plt.plot(mmd_losses, label='MMD Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('MMD Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def plot_tsne(features, labels, domain_labels, save_path):
    """绘制t-SNE特征分布图"""
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    # 源域样本
    src_mask = np.array(domain_labels) == 'source'
    scatter = plt.scatter(reduced[src_mask, 0], reduced[src_mask, 1], 
                         c=labels[src_mask], cmap='tab10', alpha=0.6, label='Source')
    # 目标域样本
    tgt_mask = np.array(domain_labels) == 'target'
    plt.scatter(reduced[tgt_mask, 0], reduced[tgt_mask, 1], 
                c='gray', alpha=0.6, marker='x', label='Target')
    
    plt.legend()
    plt.colorbar(scatter, label='Class')
    plt.title('Feature Distribution (t-SNE)')
    plt.savefig(save_path)
    plt.close()