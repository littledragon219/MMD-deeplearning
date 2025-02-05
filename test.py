import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from sklearn.decomposition import PCA
from visualization import plot_confusion_matrix, plot_tsne  # 请确保这些函数能处理 NumPy 数据
from globalconfig import MODEL_CONFIG, TRAIN_CONFIG, CLASS_MAP  # MODEL_CONFIG、TRAIN_CONFIG 等配置

# 将项目根目录添加到Python路径（根据实际情况修改）
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))
src_dir = os.path.join(str(project_root), 'src')
sys.path.append(src_dir)

def load_excel_data(excel_path):
    """
    从 Excel 文件中加载数据，返回 X 和 y。
    假设特征列以 "d" 开头，标签列名为 "label"。
    """
    df = pd.read_excel(excel_path)
    # 提取特征列（按列名中含有 "d"）
    feature_columns = [col for col in df.columns if col.startswith("d")]
    X = df[feature_columns].values
    # 标签直接读取
    y = df["label"].values
    return X, y

def prepare_tensor(X):
    """
    将 NumPy 数组转换为 PyTorch 张量，并添加 channel 维度。
    假设 X 的形状为 (N, WINDOW_SIZE) ，转换后形状为 (N, 1, WINDOW_SIZE)
    """
    tensor = torch.tensor(X, dtype=torch.float32)
    # 若数据为二维，增加一个通道维度
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    return tensor

def test_model():
    # -------------------------------
    # 1. 加载训练好的模型
    # -------------------------------
    device = torch.device(TRAIN_CONFIG["device"])
    # 根据配置构造模型结构
    from models.CNN import BearingCNN  # 模型定义在 models/CNN.py 中
    model = BearingCNN(**MODEL_CONFIG).to(device)
    
    model_dir = Path("results/models")
    best_model_path = model_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model file not found: {best_model_path}")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {best_model_path}")

    # -------------------------------
    # 2. 加载测试数据
    # -------------------------------
    processed_dir = Path("D:/大三上学期/科研训练2.0 MMD算法/input/preprocessed")
    # 无标签版本（仅用于提取样本顺序）
    target_test_file = processed_dir / "target_test.xlsx"
    # 带标签版本（真实标签）
    target_labelled_file = processed_dir / "target_labelledtest.xlsx"

    if not target_test_file.exists() or not target_labelled_file.exists():
        raise FileNotFoundError("One or both target Excel files not found.")
    
    # 从无标签版加载样本（主要获得 X 数据），从带标签版加载真实标签
    X_unlab, _ = load_excel_data(target_test_file)
    X_labelled, y_true = load_excel_data(target_labelled_file)
    
    # 检查两个文件样本数是否一致
    if X_unlab.shape[0] != X_labelled.shape[0]:
        raise ValueError("Mismatch between target test and labelled test sample numbers.")
    print(f"Loaded {X_unlab.shape[0]} target samples.")

    # 使用其中任意一个作为输入（样本顺序应一致）
    X = X_unlab  # 或 X_labelled
    # 转换为 tensor
    X_tensor = prepare_tensor(X).to(device)

    # -------------------------------
    # 3. 进行预测
    # -------------------------------
    batch_size = TRAIN_CONFIG["batch_size"]
    predictions = []
    with torch.no_grad():
        for i in range(0, X_tensor.size(0), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    predictions = np.array(predictions)
    print(f"Predictions obtained, first 20 predictions: {predictions[:20]}")

    # -------------------------------
    # 4. 计算准确率
    # -------------------------------
    # 真实标签应为数字（与 CLASS_MAP 中的数值一致）
    y_true = np.array(y_true)
    correct = (predictions == y_true).sum()
    accuracy = correct / len(y_true)
    print(f"Accuracy on target domain: {accuracy*100:.2f}% ( {correct} / {len(y_true)} )")

    # -------------------------------
    # 5. 可视化：绘制混淆矩阵和 t-SNE 分布图
    # -------------------------------
    # 绘制混淆矩阵
    class_names = [str(i) for i in sorted(set(y_true))]
    plot_confusion_matrix(y_true, predictions, class_names, "results/figures/target_confusion_matrix.png")
    
    # 提取特征进行 t-SNE 可视化
    # 这里使用模型最后一层的输出作为特征
    all_feats = []
    with torch.no_grad():
        for i in range(0, X_tensor.size(0), batch_size):
            batch = X_tensor[i:i+batch_size]
            # 假设 get_layer_outputs 返回列表，其中最后一项为所需特征
            feats = model.get_layer_outputs(batch)[-1]
            all_feats.append(feats.cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    # 如果特征是多维（如三维），将其展平为二维
    all_feats_2d = all_feats.reshape(all_feats.shape[0], -1)
    # 使用 PCA 降维到 50 维
    pca = PCA(n_components=50)
    feats_pca = pca.fit_transform(all_feats_2d)
    # 利用 t-SNE 将 50 维特征映射到 2D 空间
    # 这里我们将真实标签作为颜色标记，同时标注域信息
    domains = np.array(["target"] * len(y_true))
    plot_tsne(feats_pca, y_true, domains, "results/figures/target_tsne_distribution.png")

if __name__ == "__main__":
    test_model()
