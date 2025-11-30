# src/models/baseline.py
import torch.nn as nn

class SOTABaseline(nn.Module):
    """
    代表目前的 SOTA 方法 (如 Frozen CLIP + MLP Head)。
    没有记忆模块，仅依赖参数化知识。
    """
    def __init__(self, input_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Binary Classification
        )
        
    def forward(self, x):
        return self.classifier(x)