"""
文件：CNN.py
功能：复现论文中模型参数、MMD配置、训练超参数
作者：weng
日期：2025-02-03

说明：
    1. 可扩展性
    2.由于globalconfig.py和models不在同一个文件夹，所以需要先得到src.path，才能from models.CNN

"""
import os
from pathlib import Path
import torch
import sys

# 获取当前文件（globalconfig.py）的绝对路径
current_dir = Path(__file__).parent.absolute()

# 添加src目录到系统路径
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))
from models.CNN import BearingCNN
from models.losses import MultiKernelMMD
# === 路径配置 ===
# 原始数据目录
RAW_DATA_DIR =  "D:/大三上学期/科研训练2.0 MMD算法/input/raw"    

# 预处理后数据目录
PROCESSED_DIR = "D:/大三上学期/科研训练2.0 MMD算法/input/preprocessed" 

# === 预处理参数 ===
WINDOW_SIZE = 500       # 滑动窗口大小
STRIDE = WINDOW_SIZE            # 滑动步长 ,两者相等以避免数据重叠
SOURCE_LOAD = "0hp"     # 源域负载
TARGET_LOAD = "3hp"     # 目标域负载
TEST_SIZE = 0.2         # 验证集比例
RANDOM_SEED = 42        # 随机种子
CLASS_MAP = {
    "normal0":0,
    "normal3": 0,
    "innerace007": 1,
    "innerace014": 2,
    "innerace021": 3,
    "ball007": 4,
    "ball014": 5,
    "ball021": 6,
    "outerrace007": 7,
    "outerrace014": 8,
    "outerrace021": 9

}
FAULT_SIZES = ["007", "014", "021"]      # 故障尺寸列表
# === 模型参数 ===
# CNN模型配置
MODEL_CONFIG = {
    "in_channels": 1,
    "num_classes": 10,
    "conv_filters": [10, 10, 10, 10],
    "kernel_sizes": [10, 10, 10, 10],
    "fc_units": 256,
    "dropout_rate": 0.5
}

# MMD损失配置
MMD_CONFIG = {
    "kernel_scales": [1.0, 2.0, 4.0, 8.0, 16.0],
    "kernel_weights": None  # 等权重
}

# === 训练参数 ===
TRAIN_CONFIG = {
    "batch_size": 1000,
    "epochs": 3,
    "learning_rate": 1e-3,
    "mmd_weight": 1.0,      # MMD损失权重
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# === 工具函数 ===
def build_model() -> "BearingCNN":
    from models.CNN import BearingCNN  # 延迟导入，避免循环依赖
    return BearingCNN(**MODEL_CONFIG)

def build_mmd_loss() -> "MultiKernelMMD":
    from models.losses import MultiKernelMMD
    return MultiKernelMMD(**MMD_CONFIG)