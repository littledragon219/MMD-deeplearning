import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.CNN import BearingCNN
from models.losses import MultiKernelMMD
from globalconfig import build_model, build_mmd_loss, TRAIN_CONFIG, CLASS_MAP
from data_loader import load_source_data, load_target_data  # 需根据数据预处理实现
import sys
from pathlib import Path
import os
import numpy as np
from visualization import plot_curves, plot_confusion_matrix, plot_tsne
from sklearn.decomposition import PCA  # 导入 PCA

# 获取 train.py 文件所在的根目录
project_root = Path(__file__).parent.resolve()
# 将项目根目录添加到Python路径
sys.path.append(str(project_root))

# 构建 src 文件夹的路径
src_dir = os.path.join(str(project_root), 'src')
# 将 src 文件夹路径添加到 sys.path 中
sys.path.append(src_dir)

def train():
    # 初始化
    device = torch.device(TRAIN_CONFIG["device"])
    model = build_model().to(device)
    mmd_loss_fn = build_mmd_loss().to(device)
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])

    # 加载数据（需实现）
    source_train_loader, source_val_loader = load_source_data(TRAIN_CONFIG["batch_size"])
    target_test_loader = load_target_data(TRAIN_CONFIG["batch_size"])

    # 初始化记录列表
    train_losses = []
    val_losses = []
    val_accuracies = []
    mmd_losses = []

    # 初始化最佳准确率和最佳模型路径
    best_val_acc = 0.0
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_model.pth"

    # 训练循环
    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        epoch_train_loss = 0.0
        epoch_mmd_loss = 0.0
        for batch_idx, (src_data, src_labels) in enumerate(source_train_loader):
            src_data = src_data.to(device)
            src_labels = src_labels.to(device)

            # 随机采样目标域批次
            tgt_data, _ = next(iter(target_test_loader))
            tgt_data = tgt_data.to(device)

            optimizer.zero_grad()

            # 前向传播
            src_output = model(src_data)
            src_features = model.get_layer_outputs(src_data)
            tgt_features = model.get_layer_outputs(tgt_data)

            # 计算损失
            ce_loss = ce_loss_fn(src_output, src_labels)
            mmd_loss = mmd_loss_fn(src_features, tgt_features)
            total_loss = ce_loss + TRAIN_CONFIG["mmd_weight"] * mmd_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 记录批次损失
            epoch_train_loss += total_loss.item()
            epoch_mmd_loss += mmd_loss.item()

        # 计算 epoch 平均损失
        epoch_train_loss /= len(source_train_loader)
        epoch_mmd_loss /= len(source_train_loader)
        train_losses.append(epoch_train_loss)
        mmd_losses.append(epoch_mmd_loss)

        # 验证步骤（仅使用源域验证集计算准确率）
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in source_val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                val_loss += ce_loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(source_val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 打印进度
        print(f"Epoch {epoch + 1}/{TRAIN_CONFIG['epochs']} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"MMD Loss: {epoch_mmd_loss:.4f}")

        # 如果当前验证集准确率高于最佳准确率，保存当前模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"最佳模型已保存到: {best_model_path}，验证集准确率: {best_val_acc:.2f}%")

    # ==== 训练后可视化 ====
    # 创建结果目录
    os.makedirs("results/figures", exist_ok=True)

    # 1. 绘制训练曲线
    plot_curves(train_losses, val_losses, val_accuracies, mmd_losses, "results/figures")

    # 2. 绘制混淆矩阵（在验证集上）
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in source_val_loader:  # 使用验证集或单独测试集
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = [str(i) for i in CLASS_MAP.values()]  # 根据 CLASS_MAP
    plot_confusion_matrix(all_labels, all_preds, class_names, "results/figures/confusion_matrix.png")
    
    # 3. 绘制 t-SNE 特征分布
    def extract_features(loader, domain):
        features = []
        labels_list = []
        domain_list = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                # 取最后一层特征（例如 FC 层输出）
                feats = model.get_layer_outputs(inputs)[-1]
                features.append(feats.cpu().numpy())
                labels_list.append(labels.numpy())
                domain_list.append([domain] * len(labels))
        return np.concatenate(features), np.concatenate(labels_list), np.concatenate(domain_list)
    
    # 提取源域和目标域特征
    src_feats, src_labels, src_domains = extract_features(source_val_loader, 'source')
    tgt_feats, tgt_labels, tgt_domains = extract_features(target_test_loader, 'target')
    
    all_feats = np.concatenate([src_feats, tgt_feats])
    all_labels = np.concatenate([src_labels, tgt_labels])
    all_domains = np.concatenate([src_domains, tgt_domains])
    
    # 将可能为三维的特征数组转换为二维数组
    all_feats_2d = all_feats.reshape(all_feats.shape[0], -1)
    
    # 使用 PCA 将特征降维至 50 维
    pca = PCA(n_components=50)
    all_feats_pca = pca.fit_transform(all_feats_2d)
    
    # 利用 t-SNE 将 50 维特征映射到 2D 空间
    plot_tsne(all_feats_pca, all_labels, all_domains, "results/figures/tsne_distribution.png")

def evaluate(model: BearingCNN, data_loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    train()
