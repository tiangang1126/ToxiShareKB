# src/core/memory.py
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Tuple

class TopologicalMemory:
    """
    创新点 I: 拓扑记忆库 (Topological Memory)
    负责数据的聚类存储和双向对比检索。
    """
    def __init__(self, feature_dim, n_clusters=10):
        self.feature_dim = feature_dim
        self.n_clusters = n_clusters
        self.memory = []
        self.centroids = None
        self.is_built = False

    def build(self, experiences: List):
        """构建拓扑结构 (K-Means Clustering)"""
        self.memory = experiences
        features = np.stack([e.feature_m.numpy() for e in experiences])
        
        print(f"[Memory] Building topology with {self.n_clusters} clusters...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(features)
        
        self.centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        # 为每个经验分配 Cluster ID
        labels = kmeans.labels_
        for i, exp in enumerate(self.memory):
            exp.cluster_id = labels[i]
            
        self.is_built = True

    def retrieve_dual_contrastive(self, query: torch.Tensor, top_k=3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创新点 II (Retrieval Part): 双向对比检索
        返回: (Positive_Embeddings, Negative_Embeddings)
        """
        if not self.is_built:
            raise ValueError("Memory not built yet!")

        # 1. 确定 Query 所属簇 (加速检索，减少噪声)
        dists = torch.cdist(query.unsqueeze(0), self.centroids)
        cluster_idx = torch.argmin(dists).item()
        
        # 2. 仅在同簇内检索
        candidates = [e for e in self.memory if e.cluster_id == cluster_idx]
        if len(candidates) < top_k * 2: # Fallback if cluster too small
            candidates = self.memory
            
        cand_feats = torch.stack([e.feature_m for e in candidates])
        
        # 计算相似度
        sims = F.cosine_similarity(query.unsqueeze(0), cand_feats)
        
        # 3. 分离 Positive (Toxic) 和 Negative (Safe)
        # 注意：这里的 Pos/Neg 是指 Label 的类别
        toxic_indices = [i for i, e in enumerate(candidates) if e.label_t == 1]
        safe_indices = [i for i, e in enumerate(candidates) if e.label_t == 0]
        
        # Helper function to get embeddings
        def get_top_k_feats(indices, k):
            if not indices: return torch.zeros(k, self.feature_dim)
            subset_sims = sims[indices]
            # 取 Top-K
            k_actual = min(k, len(indices))
            val, idx = torch.topk(subset_sims, k_actual)
            # 对应的原始索引
            original_idx = [indices[i] for i in idx.tolist()]
            feats = torch.stack([candidates[i].feature_m for i in original_idx])
            # 如果不够K个，补零或重复 (这里简化为均值填充)
            if k_actual < k:
                padding = torch.zeros(k - k_actual, self.feature_dim)
                feats = torch.cat([feats, padding], dim=0)
            return feats

        pos_feats = get_top_k_feats(toxic_indices, top_k)
        neg_feats = get_top_k_feats(safe_indices, top_k)
        
        return pos_feats, neg_feats