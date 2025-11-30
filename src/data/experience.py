# src/data/experience.py
import torch
from dataclasses import dataclass

@dataclass
class ExperienceUnit:
    """
    ToxiShareKB 的核心数据结构 <M, T, L, R>
    用于标准化存储跨智能体的安全经验。
    """
    id: str
    feature_m: torch.Tensor  # M: 多模态特征向量 (e.g., CLIP output)
    label_t: int             # T: 标签 (0: Safe, 1: Toxic)
    logic_l: str             # L: 推理逻辑 (文本描述)
    repair_r: str            # R: 修复策略函数名
    cluster_id: int = -1     # 拓扑聚类 ID (自动分配)

    def __repr__(self):
        return f"<Exp {self.id} | Label: {self.label_t} | Cluster: {self.cluster_id}>"