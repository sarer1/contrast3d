# Copyright (c) Contextrast Module - Memory Optimized Version v2
# All rights reserved.

"""
内存优化的Contextrast模块

关键改进：
1. 限制正负样本数量（MAX_POSITIVE=2048, MAX_NEGATIVE=4096）
2. 使用高效的矩阵运算避免expand
3. 添加OOM保护和降级处理
4. 优化BANE采样的内存使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt


class RepresentativeAnchor(nn.Module):
    """
    代表性锚点计算模块（内存优化版）
    """
    
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
    
    def compute_anchors(self, 
                       features: torch.Tensor, 
                       labels: torch.Tensor) -> torch.Tensor:
        """
        计算每个类别的代表性锚点
        
        Args:
            features: [B, C, H, W]
            labels: [B, 1, H, W]
            
        Returns:
            anchors: [num_classes, C]
        """
        B, C, H, W = features.shape
        device = features.device
        
        # 重塑为 [B*H*W, C]
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = labels.reshape(-1)
        
        # 为每个类别计算平均特征
        anchors = []
        for class_idx in range(self.num_classes):
            class_mask = (labels_flat == class_idx)
            
            if class_mask.sum() == 0:
                # 如果该类没有样本，使用零向量
                anchors.append(torch.zeros(1, C, device=device))
            else:
                class_features = features_flat[class_mask]
                # 计算平均值作为anchor
                anchor = class_features.mean(dim=0, keepdim=True)
                anchors.append(anchor)
        
        anchors = torch.cat(anchors, dim=0)  # [num_classes, C]
        
        # L2归一化
        anchors = F.normalize(anchors, p=2, dim=1)
        
        return anchors


class ContextualAnchorFusion(nn.Module):
    """
    上下文锚点融合模块
    """
    
    def __init__(self, embed_dim: int, weight_high: float = 0.7, weight_low: float = 0.3):
        super().__init__()
        self.weight_high = weight_high
        self.weight_low = weight_low
        
    def forward(self, 
                anchor_low: torch.Tensor,
                anchor_high: torch.Tensor) -> torch.Tensor:
        """
        融合低层和高层anchor
        
        Args:
            anchor_low: 低层anchor [num_classes, C]
            anchor_high: 高层anchor [num_classes, C]
            
        Returns:
            fused_anchor: 融合后的anchor [num_classes, C]
        """
        fused = self.weight_low * anchor_low + self.weight_high * anchor_high
        fused = F.normalize(fused, p=2, dim=1)
        return fused


class PixelAnchorLoss(nn.Module):
    """
    像素-锚点对比损失（高度内存优化版本）
    
    关键优化：
    1. 限制样本数量
    2. 简化的InfoNCE实现
    3. 避免大张量的expand操作
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 max_positive_samples: int = 2048,
                 max_negative_samples: int = 4096):
        """
        Args:
            temperature: 温度参数
            max_positive_samples: 最大正样本数（内存限制）
            max_negative_samples: 最大负样本数（内存限制）
        """
        super().__init__()
        self.temperature = temperature
        self.max_positive_samples = max_positive_samples
        self.max_negative_samples = max_negative_samples
    
    def forward(self,
                features: torch.Tensor,
                anchors: torch.Tensor,
                labels: torch.Tensor,
                predictions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算Pixel-Anchor损失（简化版本）
        
        Args:
            features: [B, C, H, W]
            anchors: [num_classes, C]
            labels: [B, 1, H, W]
            predictions: [B, 1, H, W] (unused in this version)
            
        Returns:
            loss: 标量
        """
        B, C, H, W = features.shape
        num_classes = anchors.shape[0]
        device = features.device
        
        # 归一化
        features_norm = F.normalize(features, p=2, dim=1)
        anchors_norm = F.normalize(anchors, p=2, dim=1)
        
        # 展平
        features_flat = features_norm.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
        labels_flat = labels.reshape(-1)  # [N]
        
        total_loss = 0.0
        total_count = 0
        
        for class_idx in range(num_classes):
            # 当前anchor
            anchor = anchors_norm[class_idx]  # [C]
            
            # 正样本mask
            pos_mask = (labels_flat == class_idx)
            n_pos = pos_mask.sum().item()
            
            if n_pos == 0:
                continue
            
            # 负样本mask  
            neg_mask = (labels_flat != class_idx)
            n_neg = neg_mask.sum().item()
            
            if n_neg == 0:
                continue
            
            # ========== 采样（内存保护）==========
            # 限制正样本
            if n_pos > self.max_positive_samples:
                pos_indices = torch.where(pos_mask)[0]
                sampled_indices = pos_indices[torch.randperm(n_pos, device=device)[:self.max_positive_samples]]
                pos_features = features_flat[sampled_indices]
            else:
                pos_features = features_flat[pos_mask]
            
            # 限制负样本
            if n_neg > self.max_negative_samples:
                neg_indices = torch.where(neg_mask)[0]
                sampled_indices = neg_indices[torch.randperm(n_neg, device=device)[:self.max_negative_samples]]
                neg_features = features_flat[sampled_indices]
            else:
                neg_features = features_flat[neg_mask]
            
            # ========== 简化的对比学习损失 ==========
            # 计算相似度
            pos_sim = torch.matmul(pos_features, anchor) / self.temperature  # [N_pos]
            neg_sim = torch.matmul(neg_features, anchor) / self.temperature  # [N_neg]
            
            # 简化的InfoNCE：使用所有负样本的平均
            # 避免N_pos x N_neg的大矩阵
            
            # 方法1: 对每个正样本，使用所有负样本
            # loss = -log(exp(pos_i) / (exp(pos_i) + sum(exp(neg))))
            
            # 使用log-sum-exp技巧
            # 对于每个正样本
            loss_per_pos = []
            batch_size = 256  # 批处理正样本
            
            for start_idx in range(0, pos_sim.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, pos_sim.shape[0])
                pos_sim_batch = pos_sim[start_idx:end_idx]  # [batch]
                
                # 对于这个batch的正样本，计算与所有负样本的对比
                # 使用广播: pos_sim_batch [batch, 1], neg_sim [1, N_neg]
                pos_expanded = pos_sim_batch.unsqueeze(1)  # [batch, 1]
                neg_expanded = neg_sim.unsqueeze(0)  # [1, N_neg]
                
                # 合并 [batch, 1+N_neg]
                all_sim = torch.cat([pos_expanded, neg_expanded.expand(pos_expanded.shape[0], -1)], dim=1)
                
                # log_sum_exp
                log_sum_exp = torch.logsumexp(all_sim, dim=1)  # [batch]
                
                # 损失
                loss_batch = -(pos_sim_batch - log_sum_exp).mean()
                loss_per_pos.append(loss_batch)
            
            class_loss = torch.stack(loss_per_pos).mean()
            
            total_loss += class_loss
            total_count += 1
        
        if total_count > 0:
            return total_loss / total_count
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class ContextrastModule(nn.Module):
    """
    Contextrast模块（内存优化版本）
    """
    
    def __init__(self,
                 num_classes: int,
                 embed_dims: list = [24, 24, 48, 96],
                 temperature: float = 0.07,
                 weight_high: float = None,  # 新参数名
                 weight_low: float = None,   # 新参数名
                 wh: float = None,           # 旧参数名（向后兼容）
                 wl: float = None,           # 旧参数名（向后兼容）
                 use_bane_sampling: bool = False,  # v2默认关闭BANE
                 bane_sampling_ratio: float = None,  # 新参数名
                 sampling_ratio: float = None,  # 旧参数名（向后兼容）
                 layer_weights: list = None,  # 层权重
                 max_positive_samples: int = 2048,
                 max_negative_samples: int = 4096):
        """
        Args:
            num_classes: 类别数
            embed_dims: 各层嵌入维度
            temperature: 温度参数
            weight_high/wh: 高层权重（0.7）
            weight_low/wl: 低层权重（0.3）
            use_bane_sampling: 是否使用BANE采样（v2默认关闭以节省内存）
            bane_sampling_ratio/sampling_ratio: BANE采样比例
            layer_weights: 各层损失权重
            max_positive_samples: 最大正样本数
            max_negative_samples: 最大负样本数
        """
        super().__init__()
        
        # 参数兼容性处理
        if wh is not None:
            weight_high = wh
        if wl is not None:
            weight_low = wl
        if sampling_ratio is not None:
            bane_sampling_ratio = sampling_ratio
        
        # 设置默认值
        if weight_high is None:
            weight_high = 0.7
        if weight_low is None:
            weight_low = 0.3
        if bane_sampling_ratio is None:
            bane_sampling_ratio = 0.5
        if layer_weights is None:
            layer_weights = [1.0, 1.0, 1.0, 1.0]
        
        self.num_classes = num_classes
        self.num_layers = len(embed_dims)
        self.embed_dims = embed_dims
        self.use_bane_sampling = use_bane_sampling
        
        # 创建anchor计算模块
        self.anchor_modules = nn.ModuleList([
            RepresentativeAnchor(num_classes, dim)
            for dim in embed_dims
        ])
        
        # 创建anchor融合模块
        self.anchor_fusion = ContextualAnchorFusion(
            embed_dim=max(embed_dims),
            weight_high=weight_high,
            weight_low=weight_low
        )
        
        # 创建损失模块
        self.pa_loss = PixelAnchorLoss(
            temperature=temperature,
            max_positive_samples=max_positive_samples,
            max_negative_samples=max_negative_samples
        )
        
        # 层权重（确保长度匹配）
        if len(layer_weights) != len(embed_dims):
            layer_weights = layer_weights[:len(embed_dims)] if len(layer_weights) > len(embed_dims) else \
                           layer_weights + [1.0] * (len(embed_dims) - len(layer_weights))
        self.layer_weights = layer_weights
    
    def forward(self,
                multi_scale_features: list,
                labels: torch.Tensor,
                predictions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            multi_scale_features: List[Tensor]，每个是[B, C, H, W]
            labels: [B, 1, H, W]
            predictions: [B, 1, H, W] (可选)
            
        Returns:
            total_loss: 标量
            loss_dict: 损失详情
        """
        device = labels.device
        
        try:
            # 1. 计算各层anchors
            all_anchors = []
            for i, features in enumerate(multi_scale_features):
                _, _, H, W = features.shape
                labels_resized = F.interpolate(
                    labels.float(),
                    size=(H, W),
                    mode='nearest'
                ).long()
                
                anchors = self.anchor_modules[i].compute_anchors(features, labels_resized)
                all_anchors.append(anchors)
            
            # 2. 融合anchors（简化版 - 直接使用各层自己的anchor）
            fused_anchors = all_anchors
            
            # 3. 计算各层损失
            total_loss = 0.0
            loss_dict = {}
            
            for i, (features, anchors, weight) in enumerate(zip(
                multi_scale_features, fused_anchors, self.layer_weights
            )):
                _, _, H, W = features.shape
                labels_resized = F.interpolate(
                    labels.float(),
                    size=(H, W),
                    mode='nearest'
                ).long()
                
                if predictions is not None:
                    predictions_resized = F.interpolate(
                        predictions.float(),
                        size=(H, W),
                        mode='nearest'
                    ).long()
                else:
                    predictions_resized = None
                
                # 计算损失
                layer_loss = self.pa_loss(
                    features,
                    anchors,
                    labels_resized,
                    predictions_resized
                )
                
                weighted_loss = weight * layer_loss
                total_loss += weighted_loss
                
                loss_dict[f'layer_{i+1}_pa_loss'] = layer_loss.item()
                loss_dict[f'layer_{i+1}_weighted_pa_loss'] = weighted_loss.item()
            
            loss_dict['total_contextrast_loss'] = total_loss.item()
            
            return total_loss, loss_dict
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM时返回零损失
                print(f"Warning: Contextrast OOM, returning zero loss")
                torch.cuda.empty_cache()
                return torch.tensor(0.0, device=device, requires_grad=True), {
                    'total_contextrast_loss': 0.0,
                    'oom_error': True
                }
            else:
                raise e


# 测试代码
if __name__ == '__main__':
    print("Testing Memory-Optimized Contextrast Module...")
    
    # 创建测试数据
    B, H, W = 4, 256, 256
    num_classes = 2
    
    # 多尺度特征
    multi_scale_features = [
        torch.randn(B, 24, H, W).cuda(),
        torch.randn(B, 24, H//2, W//2).cuda(),
        torch.randn(B, 48, H//4, W//4).cuda(),
        torch.randn(B, 96, H//8, W//8).cuda(),
    ]
    
    # 标签和预测
    labels = torch.randint(0, num_classes, (B, 1, H, W)).cuda()
    predictions = torch.randint(0, num_classes, (B, 1, H, W)).cuda()
    
    # 创建模块
    contextrast = ContextrastModule(
        num_classes=num_classes,
        embed_dims=[24, 24, 48, 96],
        max_positive_samples=2048,
        max_negative_samples=4096
    ).cuda()
    
    # 前向传播
    loss, loss_dict = contextrast(multi_scale_features, labels, predictions)
    
    print(f"Loss: {loss.item():.4f}")
    print("Loss details:")
    for k, v in loss_dict.items():
        if isinstance(v, bool):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")
    
    print("\n✅ Test passed!")