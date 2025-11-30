# src/core/calibrator.py
import torch
import torch.nn as nn

class ContrastiveCalibrator(nn.Module):
    """
    创新点 II & III: 对比校准与逻辑注入模块
    """
    def __init__(self, dim=128):
        super().__init__()
        # 门控网络：决定多大程度上依赖检索结果
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3), # Weights for [Query, Pos_Neighbor, Neg_Neighbor]
            nn.Softmax(dim=-1)
        )
        
        # 投影层：用于计算特征差分
        self.calibration_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        # 逻辑注入 (Logic Injection): 简单的 Self-Attention 模拟
        self.logic_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, query, pos_mem, neg_mem):
        """
        query: [B, D]
        pos_mem: [B, K, D] (Mean pooled to [B, D] inside)
        neg_mem: [B, K, D]
        """
        # 简单均值池化 (也可以用 Attention 聚合)
        p_vec = pos_mem.mean(dim=1) # [B, D]
        n_vec = neg_mem.mean(dim=1) # [B, D]
        
        # 1. 计算动态权重
        concat_feats = torch.cat([query, p_vec, n_vec], dim=-1)
        gates = self.gate_net(concat_feats) # [B, 3]
        w_q, w_p, w_n = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        
        # 2. 对比校准公式 (The Secret Sauce)
        # 核心思想：Query 应该往 Positive 靠，往 Negative 远
        # Calibrated = Query + alpha * (Pos - Neg)
        delta = p_vec - n_vec
        
        # 这里的逻辑是：如果 Query 很模糊，利用 delta (毒性方向向量) 来推一把
        calibrated = w_q * query + (w_p + w_n) * self.calibration_proj(delta)
        
        # Residual Connection
        out = self.norm(calibrated + query)
        
        return out