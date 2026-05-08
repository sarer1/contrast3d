# Copyright (c) Contextrast-Enhanced Change3D
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from scipy.ndimage import distance_transform_edt
import numpy as np


class ContextualContrastiveLearning(nn.Module):
    """
    上下文对比学习模块
    
    从多尺度特征中提取代表性锚点，并使用高层锚点更新低层锚点
    以捕获全局和局部上下文的关系
    """
    
    def __init__(self, 
                 embed_dims: List[int] = [24, 24, 48, 96],
                 num_classes: int = 2,
                 temperature: float = 0.07,
                 wl: float = 0.3,
                 wh: float = 0.7):
        """
        Args:
            embed_dims: 各层级的嵌入维度
            num_classes: 类别数（二值变化检测为2）
            temperature: 温度参数τ
            wl: 低层锚点权重
            wh: 高层锚点权重
        """
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.temperature = temperature
        self.wl = wl
        self.wh = wh
        self.num_layers = len(embed_dims)
        
        # 为每一层创建投影头，将特征投影到统一的嵌入空间
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1, bias=False)
            ) for dim in embed_dims
        ])
        
        # 层级权重（用于加权pixel-anchor loss）
        self.register_buffer('layer_weights', 
                           torch.tensor([0.1, 0.4, 0.7, 1.0]))
        
    def compute_anchors(self, 
                       embeddings: torch.Tensor, 
                       labels: torch.Tensor) -> torch.Tensor:
        """
        计算代表性锚点（类别中心）
        
        Args:
            embeddings: [B, D, H, W] 嵌入特征
            labels: [B, 1, H, W] 标签
            
        Returns:
            anchors: [N, D] 每个类别的锚点
        """
        B, D, H, W = embeddings.shape
        
        # 展平
        embeddings_flat = embeddings.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
        labels_flat = labels.reshape(-1)  # [B*H*W]
        
        anchors = []
        for c in range(self.num_classes):
            mask = (labels_flat == c)
            if mask.sum() > 0:
                class_embeddings = embeddings_flat[mask]  # [N_c, D]
                anchor = class_embeddings.mean(dim=0)  # [D]
            else:
                # 如果类别不存在，使用零向量
                anchor = torch.zeros(D, device=embeddings.device)
            anchors.append(anchor)
        
        anchors = torch.stack(anchors, dim=0)  # [N, D]
        return anchors
    
    def update_anchors(self, 
                      anchors_low: torch.Tensor,
                      anchors_high: torch.Tensor) -> torch.Tensor:
        """
        使用高层锚点更新低层锚点
        
        Args:
            anchors_low: [N, D] 低层锚点
            anchors_high: [N, D] 高层锚点
            
        Returns:
            updated_anchors: [N, D] 更新后的锚点
        """
        return self.wl * anchors_low + self.wh * anchors_high
    
    def forward(self, 
                features: List[torch.Tensor],
                labels: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            features: 多尺度特征 [f1, f2, f3, f4]
            labels: [B, 1, H, W] 标签
            
        Returns:
            embeddings_list: 投影后的嵌入特征列表
            anchors_list: 更新后的锚点列表
        """
        embeddings_list = []
        raw_anchors_list = []
        
        # Step 1: 为每层特征计算投影和原始锚点
        for i, feat in enumerate(features):
            # 投影到嵌入空间
            embedding = self.projectors[i](feat)  # [B, 128, H, W]
            
            # 下采样标签以匹配特征尺寸
            _, _, H, W = embedding.shape
            labels_resized = F.interpolate(
                labels.float(), 
                size=(H, W), 
                mode='nearest'
            ).long()
            
            # 计算锚点
            anchors = self.compute_anchors(embedding, labels_resized)
            
            embeddings_list.append(embedding)
            raw_anchors_list.append(anchors)
        
        # Step 2: 使用最高层锚点更新所有层的锚点
        highest_anchors = raw_anchors_list[-1]
        updated_anchors_list = []
        
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                # 最高层不更新
                updated_anchors = highest_anchors
            else:
                # 低层锚点融合高层信息
                updated_anchors = self.update_anchors(
                    raw_anchors_list[i], 
                    highest_anchors
                )
            updated_anchors_list.append(updated_anchors)
        
        return embeddings_list, updated_anchors_list


class BANESampling(nn.Module):
    """
    边界感知负样本采样 (Boundary-Aware Negative Sampling)
    
    选择错误预测区域边界附近的像素作为困难负样本
    """
    
    def __init__(self, sampling_ratio: float = 0.5):
        """
        Args:
            sampling_ratio: 负样本采样比例 K%
        """
        super().__init__()
        self.sampling_ratio = sampling_ratio
    
    def compute_distance_map(self, 
                            error_map: np.ndarray) -> np.ndarray:
        """
        计算距离变换图
        
        Args:
            error_map: [H, W] 二值错误图（0/1）
            
        Returns:
            distance_map: [H, W] 距离变换图
        """
        if error_map.sum() == 0:
            return error_map.astype(np.float32)
        
        # 距离变换（计算每个像素到最近边界的距离）
        distance_map = distance_transform_edt(error_map)
        return distance_map
    
    def forward(self,
                prediction: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        生成边界感知负样本掩码
        
        Args:
            prediction: [B, 1, H, W] 预测结果（sigmoid后）
            labels: [B, 1, H, W] 真实标签
            
        Returns:
            hard_negative_mask: [B, 1, H, W] 困难负样本掩码
        """
        B, _, H, W = prediction.shape
        hard_negative_masks = []
        
        # 二值化预测
        pred_binary = (prediction > 0.5).float()
        
        for b in range(B):
            pred_b = pred_binary[b, 0].cpu().numpy()  # [H, W]
            label_b = labels[b, 0].cpu().numpy()  # [H, W]
            
            # 对每个类别计算错误图和距离图
            hard_neg_mask = np.zeros((H, W), dtype=np.float32)
            
            for c in range(2):
                # 类别c的错误预测区域
                error_map = ((pred_b != c) & (label_b == c)).astype(np.uint8)
                
                if error_map.sum() == 0:
                    continue
                
                # 计算距离图
                dist_map = self.compute_distance_map(error_map)
                
                # 选择距离最小的K%像素（边界附近）
                error_pixels = np.where(error_map > 0)
                if len(error_pixels[0]) == 0:
                    continue
                
                distances = dist_map[error_pixels]
                threshold_dist = np.percentile(distances, self.sampling_ratio * 100)
                
                # 标记为困难负样本
                hard_neg_mask[(error_map > 0) & (dist_map <= threshold_dist)] = 1.0
            
            hard_negative_masks.append(
                torch.from_numpy(hard_neg_mask).unsqueeze(0)
            )
        
        hard_negative_mask = torch.stack(hard_negative_masks, dim=0)  # [B, 1, H, W]
        return hard_negative_mask.to(prediction.device)


class PixelAnchorLoss(nn.Module):
    """
    像素-锚点对比损失
    
    拉近同类像素与其锚点，推开不同类像素与其他类锚点
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 use_bane: bool = True):
        """
        Args:
            temperature: 温度参数
            use_bane: 是否使用BANE采样
        """
        super().__init__()
        self.temperature = temperature
        self.use_bane = use_bane
        
    def forward(self,
                embeddings: torch.Tensor,
                anchors: torch.Tensor,
                labels: torch.Tensor,
                hard_negative_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算pixel-anchor对比损失
        
        Args:
            embeddings: [B, D, H, W] 嵌入特征
            anchors: [N, D] 类别锚点
            labels: [B, 1, H, W] 标签
            hard_negative_mask: [B, 1, H, W] 困难负样本掩码（可选）
            
        Returns:
            loss: 标量损失
        """
        B, D, H, W = embeddings.shape
        N = anchors.shape[0]
        
        # 归一化
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [B, D, H, W]
        anchors_norm = F.normalize(anchors, p=2, dim=1)  # [N, D]
        
        # 展平
        embeddings_flat = embeddings_norm.permute(0, 2, 3, 1).reshape(B * H * W, D)
        labels_flat = labels.reshape(B * H * W)
        
        # 计算相似度矩阵
        similarity = torch.mm(embeddings_flat, anchors_norm.t()) / self.temperature
        # [B*H*W, N]
        
        # 构造目标（one-hot）
        targets = F.one_hot(labels_flat, num_classes=N).float()  # [B*H*W, N]
        
        # 如果使用BANE采样，对困难负样本加权
        if self.use_bane and hard_negative_mask is not None:
            hard_mask_flat = hard_negative_mask.reshape(B * H * W)
            
            # 对困难负样本增加权重
            weight = torch.ones_like(labels_flat).float()
            weight[hard_mask_flat > 0] = 2.0  # 困难样本权重加倍
        else:
            weight = torch.ones_like(labels_flat).float()
        
        # InfoNCE损失
        # L = -log(exp(sim_pos) / sum(exp(sim_all)))
        exp_sim = torch.exp(similarity)  # [B*H*W, N]
        
        # 正样本相似度
        pos_sim = (exp_sim * targets).sum(dim=1)  # [B*H*W]
        
        # 所有样本相似度和
        all_sim = exp_sim.sum(dim=1)  # [B*H*W]
        
        # 损失
        loss = -torch.log(pos_sim / (all_sim + 1e-8))  # [B*H*W]
        
        # 加权平均
        loss = (loss * weight).sum() / weight.sum()
        
        return loss


class ContrastiveLearningModule(nn.Module):
    """
    完整的对比学习模块
    
    整合CCL和BANE
    """
    
    def __init__(self,
                 embed_dims: List[int] = [24, 24, 48, 96],
                 num_classes: int = 2,
                 temperature: float = 0.07,
                 alpha: float = 0.1,
                 use_bane: bool = True,
                 bane_ratio: float = 0.5):
        """
        Args:
            embed_dims: 嵌入维度列表
            num_classes: 类别数
            temperature: 温度参数
            alpha: 对比损失权重
            use_bane: 是否使用BANE
            bane_ratio: BANE采样比例
        """
        super().__init__()
        
        self.ccl = ContextualContrastiveLearning(
            embed_dims=embed_dims,
            num_classes=num_classes,
            temperature=temperature
        )
        
        if use_bane:
            self.bane = BANESampling(sampling_ratio=bane_ratio)
        else:
            self.bane = None
        
        self.pa_loss = PixelAnchorLoss(
            temperature=temperature,
            use_bane=use_bane
        )
        
        self.alpha = alpha
        self.use_bane = use_bane
        
    def forward(self,
                features: List[torch.Tensor],
                prediction: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            features: 多尺度特征列表
            prediction: [B, 1, H, W] 预测（sigmoid后）
            labels: [B, 1, H, W] 标签
            
        Returns:
            contrastive_loss: 对比损失
            info_dict: 信息字典
        """
        # CCL: 获取嵌入和锚点
        embeddings_list, anchors_list = self.ccl(features, labels)
        
        # BANE: 生成困难负样本掩码
        if self.use_bane:
            hard_negative_mask = self.bane(prediction, labels)
        else:
            hard_negative_mask = None
        
        # 计算多尺度pixel-anchor损失
        total_loss = 0
        layer_losses = []
        
        for i, (embedding, anchors) in enumerate(zip(embeddings_list, anchors_list)):
            # 调整标签尺寸
            _, _, H, W = embedding.shape
            labels_resized = F.interpolate(
                labels.float(),
                size=(H, W),
                mode='nearest'
            ).long()
            
            # 调整困难负样本掩码尺寸
            if hard_negative_mask is not None:
                mask_resized = F.interpolate(
                    hard_negative_mask.float(),
                    size=(H, W),
                    mode='nearest'
                )
            else:
                mask_resized = None
            
            # 计算损失
            layer_loss = self.pa_loss(
                embedding,
                anchors,
                labels_resized,
                mask_resized
            )
            
            # 加权
            weighted_loss = self.ccl.layer_weights[i] * layer_loss
            total_loss += weighted_loss
            layer_losses.append(layer_loss.item())
        
        # 平均
        contrastive_loss = total_loss / len(embeddings_list)
        
        info_dict = {
            'contrastive_loss': contrastive_loss.item(),
            'layer_losses': layer_losses,
        }
        
        return contrastive_loss, info_dict