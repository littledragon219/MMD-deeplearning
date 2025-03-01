import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from typing import List, Optional

# === 预处理参数 ===
WINDOW_SIZE = 500      # 滑动窗口大小
STRIDE = 500           # 滑动步长，两者相等以避免数据重叠
SOURCE_LOAD = "0hp"    # 源域负载
TARGET_LOAD = "3hp"    # 目标域负载
TEST_SIZE = 0.2        # 验证集比例
RANDOM_SEED = 42       # 随机种子
CLASS_MAP = {
    "normal0": 0,
    "normal3": 0,
    "innerace007": 1,
    "ball007": 2,
    "outerrace007": 3,
    "innerace014": 4,
    "ball014": 5,
    "outerrace014": 6,
}

# === 模型参数 ===
MODEL_CONFIG = {
    "in_channels": 1,
    "num_classes": 10,
    "conv_filters": [10, 10, 10, 10],
    "kernel_sizes": [10, 10, 10, 10],
    "fc_units": 256,
    "dropout_rate": 0.5
}

MMD_CONFIG = {
    "kernel_scales": [1.0, 2.0, 4.0, 8.0, 16.0],
    "kernel_weights": None  # 等权重
}

# === 训练参数 ===
TRAIN_CONFIG = {
    "batch_size": 500,
    "epochs": 2000,
    "learning_rate": 1e-3,
    "mmd_weight": 1.0,  # MMD损失权重
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 定义预处理数据所在目录（请根据实际情况修改）
PROCESSED_DIR = "processed_data"

# =======================
# 定义标签平滑交叉熵损失函数
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# =======================
# 定义 4 层 1D CNN 模型
class BearingCNN(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 conv_filters: List[int] = [10, 10, 10, 10],
                 kernel_sizes: List[int] = [10, 10, 10, 10],
                 fc_units: int = 256,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for filters, kernel_size in zip(conv_filters, kernel_sizes):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, kernel_size, padding="same"),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Conv1d(filters, filters, 3, padding=1),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(0.3)
                )
            )
            in_channels = filters

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self._calculate_fc_input(), fc_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _calculate_fc_input(self) -> int:
        dummy = torch.zeros(1, 1, WINDOW_SIZE)
        with torch.no_grad():
            for layer in self.conv_layers:
                dummy = layer(dummy)
        return dummy.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def get_layer_outputs(self, x: torch.Tensor, layers: Optional[List[int]] = None) -> List[torch.Tensor]:
        if layers is None:
            layers = list(range(len(self.conv_layers)))
        outputs = []
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i in layers:
                outputs.append(x)
        return outputs

# =======================
# 定义多核 MMD 损失
class MultiKernelMMD(nn.Module):
    def __init__(self,
                 kernel_scales: List[float] = [1.0, 2.0, 4.0, 8.0, 16.0],
                 kernel_weights: Optional[List[float]] = None):
        super().__init__()
        self.kernel_scales = kernel_scales
        self.kernel_weights = kernel_weights or [1.0 / len(kernel_scales)] * len(kernel_scales)

    def forward(self, source_features: List[torch.Tensor],
                target_features: List[torch.Tensor]) -> torch.Tensor:
        if len(source_features) != len(target_features):
            raise ValueError("Source and target must have same number of feature layers")
        total_mmd = 0.0
        for src_feat, tgt_feat in zip(source_features, target_features):
            src = src_feat.view(src_feat.size(0), -1)
            tgt = tgt_feat.view(tgt_feat.size(0), -1)
            layer_mmd = 0.0
            for scale, weight in zip(self.kernel_scales, self.kernel_weights):
                gamma = 1.0 / (2 * scale ** 2)
                layer_mmd += weight * self._rbf_mmd(src, tgt, gamma)
            total_mmd += layer_mmd
        return total_mmd

    def _rbf_mmd(self, X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
        XX = torch.matmul(X, X.T)
        XY = torch.matmul(X, Y.T)
        YY = torch.matmul(Y, Y.T)
        X_sqnorms = torch.diag(XX)
        Y_sqnorms = torch.diag(YY)
        K_XX = torch.exp(-gamma * (X_sqnorms.unsqueeze(1) - 2 * XX + X_sqnorms.unsqueeze(0)))
        K_XY = torch.exp(-gamma * (X_sqnorms.unsqueeze(1) - 2 * XY + Y_sqnorms.unsqueeze(0)))
        K_YY = torch.exp(-gamma * (Y_sqnorms.unsqueeze(1) - 2 * YY + Y_sqnorms.unsqueeze(0)))
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return mmd

# =======================
# 定义数据集类（加载 Excel 数据）
class BearingDataset(Dataset):
    def __init__(self, excel_path: str, is_source: bool = True):
        df = pd.read_excel(excel_path)
        # 提取特征（假设前 N-1 列为特征，最后一列为标签）
        self.features = df.iloc[:, :-1].values.astype(np.float32)
        self.features = torch.from_numpy(self.features).unsqueeze(1)
        self.is_source = is_source
        if self.is_source:
            try:
                labels = df["label"].values.astype(np.int64)
            except ValueError:
                labels = df["label"].apply(
                    lambda x: int(x) if str(x).isdigit() else CLASS_MAP.get(x, -1)
                ).values
            self.labels = torch.from_numpy(labels)
        else:
            # 如果不是源数据，依然尝试加载标签（可能包含有效类别）
            try:
                labels = df["label"].values.astype(np.int64)
            except ValueError:
                labels = df["label"].apply(
                    lambda x: int(x) if str(x).isdigit() else CLASS_MAP.get(x, -1)
                ).values
            self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# =======================
# 构造模型和损失函数
def build_model() -> BearingCNN:
    return BearingCNN(**MODEL_CONFIG)

def build_mmd_loss() -> MultiKernelMMD:
    return MultiKernelMMD(**MMD_CONFIG)

# =======================
# 数据加载函数
def load_source_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataset = BearingDataset(
        os.path.join(PROCESSED_DIR, "/kaggle/input/6samples/source_train.xlsx"),
        is_source=True
    )
    val_dataset = BearingDataset(
        os.path.join(PROCESSED_DIR, "/kaggle/input/6samples/source_val.xlsx"),
        is_source=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def load_target_data(batch_size: int) -> DataLoader:
    """加载目标数据（测试集）：使用 target_test.xlsx"""
    test_dataset = BearingDataset(
        os.path.join(PROCESSED_DIR, "/kaggle/input/6samples/target_test.xlsx"),
        is_source=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

def load_target_dataa(batch_size: int) -> DataLoader:
    """加载目标标注数据（测试集）：使用 target_labelledtest.xlsx"""
    test_dataset = BearingDataset(
        os.path.join(PROCESSED_DIR, "/kaggle/input/6samples/target_labelledtest.xlsx"),
        is_source=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

# =======================
# 特征提取和 TSNE 可视化函数
def extract_features(loader, model, device, domain):
    features = []
    labels_list = []
    domain_list = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            if isinstance(model, nn.DataParallel):
                layer_outputs = model.module.get_layer_outputs(inputs)
            else:
                layer_outputs = model.get_layer_outputs(inputs)
            feats = layer_outputs[-1]  # 取最后一层卷积输出
            features.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            domain_list.append([domain] * len(labels))
    return np.concatenate(features), np.concatenate(labels_list), np.concatenate(domain_list)

def plot_tsne(features, labels, domains, save_path):
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    embeddings = tsne.fit_transform(features)
    plt.figure(figsize=(12, 6))
    
    # 子图1：按类别区分（仅颜色区分）
    plt.subplot(121)
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1],
                    hue=labels, palette="tab10", s=100)
    plt.title("Feature Distribution by Class")
    
    # 子图2：按数据来源区分（颜色和 marker 均区分）
    plt.subplot(122)
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1],
                    hue=domains, style=domains, palette="Set2", s=100)
    plt.title("Feature Distribution by Domain")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close()

# =======================
# 训练函数
def train():
    device = torch.device(TRAIN_CONFIG["device"])
    model = build_model().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    mmd_loss_fn = build_mmd_loss().to(device)
    ce_loss_fn = LabelSmoothingCE(smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"], weight_decay=1e-4)
    source_train_loader, source_val_loader = load_source_data(TRAIN_CONFIG["batch_size"])
    target_test_loader = load_target_data(TRAIN_CONFIG["batch_size"])

    best_val_acc = 0.0
    model_dir = Path("/kaggle/working/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_modelnew.pth"

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        epoch_train_loss = 0.0
        epoch_mmd_loss = 0.0
        target_iter = iter(target_test_loader)
        for batch_idx, (src_data, src_labels) in enumerate(source_train_loader):
            src_data = src_data.to(device)
            src_labels = src_labels.to(device)
            try:
                tgt_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_test_loader)
                tgt_data, _ = next(target_iter)
            tgt_data = tgt_data.to(device)
            optimizer.zero_grad()
            if isinstance(model, nn.DataParallel):
                src_output = model(src_data)
                src_features = model.module.get_layer_outputs(src_data)
                tgt_features = model.module.get_layer_outputs(tgt_data)
            else:
                src_output = model(src_data)
                src_features = model.get_layer_outputs(src_data)
                tgt_features = model.get_layer_outputs(tgt_data)
            ce_loss = ce_loss_fn(src_output, src_labels)
            mmd_loss = mmd_loss_fn(src_features, tgt_features)
            current_progress = (epoch + batch_idx / len(source_train_loader)) / TRAIN_CONFIG["epochs"]
            mmd_weight = 2.0 * (1 - current_progress)
            total_loss = ce_loss + mmd_weight * mmd_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_loss.backward()
            optimizer.step()
            epoch_train_loss += total_loss.item()
            epoch_mmd_loss += mmd_loss.item()

        epoch_train_loss /= len(source_train_loader)
        epoch_mmd_loss /= len(source_train_loader)
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
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']} | Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | MMD Loss: {epoch_mmd_loss:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to: {best_model_path} with Val Acc: {best_val_acc:.2f}%")

    os.makedirs("results/figures", exist_ok=True)
    src_feats, src_labels, src_domains = extract_features(source_val_loader, model, device, 'source')
    tgt_feats, tgt_labels, tgt_domains = extract_features(target_test_loader, model, device, 'target_test')
    all_feats = np.concatenate([src_feats, tgt_feats])
    all_labels = np.concatenate([src_labels, tgt_labels])
    all_domains = np.concatenate([src_domains, tgt_domains])
    all_feats_2d = all_feats.reshape(all_feats.shape[0], -1)
    pca = PCA(n_components=50)
    all_feats_pca = pca.fit_transform(all_feats_2d)
    plot_tsne(all_feats_pca, all_labels, all_domains, "/kaggle/working/tsne_distribution.png")

# =======================
# 评估函数（计算准确率）
def evaluate(model: BearingCNN, data_loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# =======================
# 利用最佳模型在两个目标数据集上评估并生成 TSNE 图
def evaluate_best_model_on_target():
    device = torch.device(TRAIN_CONFIG["device"])
    model = build_model().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    best_model_path = Path("/kaggle/working/models") / "best_modelnew.pth"
    try:
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(best_model_path))
        else:
            model.load_state_dict(torch.load(best_model_path))
    except FileNotFoundError:
        print(f"Error: Model file {best_model_path} not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        # 分别加载两个目标数据集
        target_test_loader = load_target_data(TRAIN_CONFIG["batch_size"])
        target_labelled_loader = load_target_dataa(TRAIN_CONFIG["batch_size"])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    model.eval()
    all_predictions = []
    all_labels = []

    # 对 target_test 进行模型验证并预测类别
    with torch.no_grad():
        for inputs, _ in target_test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().tolist())

    # 获取 target_labelled 数据集的真实标签
    with torch.no_grad():
        for _, labels in target_labelled_loader:
            all_labels.extend(labels.cpu().tolist())

    # 确保两个数据集的样本数量一致
    if len(all_predictions) != len(all_labels):
        print("Error: The number of samples in target_test and target_labelled must be the same.")
        return

    # 计算准确率
    correct = sum([p == l for p, l in zip(all_predictions, all_labels)])
    accuracy = correct / len(all_predictions) * 100
    print(f"Accuracy (Resolution): {accuracy:.2f}%")

 

    # 分别提取两个数据集的特征
    feats_test, labels_test, domains_test = extract_features(target_test_loader, model, device, 'target_test')
    feats_labelled, labels_labelled, domains_labelled = extract_features(target_labelled_loader, model, device, 'target_labelledtest')
    
    # 合并两个数据集的特征、标签和数据来源信息
    all_feats = np.concatenate([feats_test, feats_labelled])
    all_labels = np.concatenate([labels_test, labels_labelled])
    all_domains = np.concatenate([domains_test, domains_labelled])
    
    # 先将特征展平，再用 PCA 降维，再用 TSNE 降为2维
    all_feats_2d = all_feats.reshape(all_feats.shape[0], -1)
    pca = PCA(n_components=50)
    all_feats_pca = pca.fit_transform(all_feats_2d)
    
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    embeddings = tsne.fit_transform(all_feats_pca)
    
    # --- TSNE 绘图 ---
    # 为了同时区分类别和数据来源：
    #   - 用 hue 表示类别（class），即 all_labels
    #   - 用 marker 形状表示数据来源（domain），即 all_domains
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=all_labels,          # 颜色区分类别
        style=all_domains,       # marker 形状区分数据来源
        palette="tab10",         # 可根据类别数目选择调色板
        s=100
    )
    plt.title("TSNE: target_test vs target_labelledtest\n(Color: Class, Marker: Domain)")
    plt.tight_layout()
    tsne_save_path = "/kaggle/working/tsne_combined_targets.png"
    plt.savefig(tsne_save_path)
    plt.clf()
    plt.close()
    print(f"Combined TSNE plot saved to: {tsne_save_path}")

# =======================
# 主程序入口
if __name__ == "__main__":
    train()
    evaluate_best_model_on_target()