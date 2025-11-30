# src/models/toxishare.py
import torch
import torch.nn as nn
from ..core.memory import TopologicalMemory
from ..core.calibrator import ContrastiveCalibrator

class ToxiShareNetwork(nn.Module):
    def __init__(self, input_dim=128, kb_data=None):
        super().__init__()
        # 1. 记忆库
        self.memory = TopologicalMemory(feature_dim=input_dim)
        if kb_data:
            self.memory.build(kb_data)
            
        # 2. 核心组件
        self.calibrator = ContrastiveCalibrator(dim=input_dim)
        
        # 3. 分类头 (简单线性，证明提升来自校准而非分类器深度)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        """
        x: [B, D] input features
        """
        batch_size = x.size(0)
        calibrated_features = []
        
        # 逐样本检索 (Batch 优化版本在实际工程中需用 FAISS GPU)
        # 这里为了演示清晰使用循环
        for i in range(batch_size):
            curr_x = x[i]
            
            # Step A: 检索
            # 同时拿到“有毒邻居”和“安全邻居”
            p_mem, n_mem = self.memory.retrieve_dual_contrastive(curr_x, top_k=5)
            
            # Step B: 校准
            # 将邻居信息作为 Context 传入
            calib_feat = self.calibrator(curr_x.unsqueeze(0), 
                                         p_mem.unsqueeze(0), 
                                         n_mem.unsqueeze(0))
            calibrated_features.append(calib_feat)
            
        # 堆叠回 Batch
        final_input = torch.cat(calibrated_features, dim=0)
        
        # Step C: 分类
        logits = self.classifier(final_input)
        return logits