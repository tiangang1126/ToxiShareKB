# src/data/mock_data.py
import torch
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from .experience import ExperienceUnit

def generate_complex_toxicity_data(n_samples=2000, feat_dim=128):
    """
    生成模拟的 Embedding 数据，包含“隐蔽毒性”特征。
    使用 make_moons 模拟非线性决策边界 (SOTA 线性探针的弱点)。
    """
    print(f"[Data] Generating synthetic hard examples (Non-linear boundary)...")
    
    # 1. 生成基础分布 (2D流形映射到高维)
    X_2d, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    
    # 2. 映射到高维特征空间 (模拟 BERT/CLIP 输出)
    # 我们用随机正交矩阵将 2D 投影到 feat_dim 维度
    projection = np.random.randn(2, feat_dim)
    X_high_dim = np.dot(X_2d, projection)
    
    # 3. 添加噪声 (模拟现实世界的 messy data)
    X_high_dim += np.random.normal(0, 0.1, size=X_high_dim.shape)
    
    # 转换为 Tensor
    X_tensor = torch.tensor(X_high_dim, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # 4. 构建经验库数据集 (Seed Knowledge Base)
    # 假设训练集就是我们的“历史经验”
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    knowledge_base_data = []
    for i in range(len(X_train)):
        # 模拟生成 Logic 和 Repair 字段
        is_toxic = y_train[i].item() == 1
        logic = "Visual gun detected" if is_toxic else "Artistic context verified"
        repair = "pixel_masking" if is_toxic else "none"
        
        exp = ExperienceUnit(
            id=f"train_{i}",
            feature_m=X_train[i],
            label_t=y_train[i].item(),
            logic_l=logic,
            repair_r=repair
        )
        knowledge_base_data.append(exp)
        
    return knowledge_base_data, X_test, y_test