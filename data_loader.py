"""
文件：data_loader.py
功能：读取预处理好的数据
作者：weng
日期：2025-02-03

说明：
1:数据格式转换

Excel中的每行数据被转换为 (1, 500) 的Tensor，符合1D CNN的输入要求（通道数=1）。

2.标签处理

源域标签转换为 torch.int64，目标域标签设为 -1（表示无标签）。

3.性能优化

使用 pin_memory=True 加速GPU数据传输。

4.多线程加载（num_workers=4）。

与训练脚本集成
5.在 train.py 中调用：

"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from globalconfig import PROCESSED_DIR, TRAIN_CONFIG

class BearingDataset(Dataset):
    """加载预处理后的Excel文件，生成PyTorch Dataset"""
    
    def __init__(self, excel_path: str, is_source: bool = True):
        """
        :param excel_path: Excel文件路径（如 'processed_data/source_train.xlsx'）
        :param is_source: 是否为源域数据（决定是否加载标签）
        """
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        
        # 提取特征（前500列为数据点，列名 d0-d499）
        self.features = df.iloc[:, :-1].values.astype(np.float32)
        # 转换为Tensor并增加通道维度 (N, 1, 500)
        self.features = torch.from_numpy(self.features).unsqueeze(1)
        
        # 处理标签
        self.is_source = is_source
        if self.is_source:
            # 源域：标签为整数
            self.labels = df["label"].values.astype(np.int64)
            self.labels = torch.from_numpy(self.labels)
        else:
            # 目标域：标签设为-1（占位符）
            self.labels = torch.full((len(self.features),), -1, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_source_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """加载源域数据（训练集和验证集）"""
    train_dataset = BearingDataset(
        os.path.join(PROCESSED_DIR, "source_train.xlsx"), 
        is_source=True
    )
    val_dataset = BearingDataset(
        os.path.join(PROCESSED_DIR, "source_val.xlsx"), 
        is_source=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 加速GPU数据传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader

def load_target_data(batch_size: int) -> DataLoader:
    """加载目标域数据（测试集）"""
    test_dataset = BearingDataset(
        os.path.join(PROCESSED_DIR, "target_test.xlsx"), 
        is_source=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader

# 示例用法（需在config.py中设置PROCESSED_DIR路径）
if __name__ == "__main__":
    train_loader, val_loader = load_source_data(64)
    test_loader = load_target_data(64)
    
    # 检查数据形状
    sample, label = next(iter(train_loader))
    print(f"训练集样本形状: {sample.shape}")  # 应为 (batch_size, 1, 500)
    print(f"训练集标签形状: {label.shape}")   # 应为 (batch_size,)