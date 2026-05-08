# Copyright (c) BCD Contrastive Learning Module
# All rights reserved.

"""
针对二值变化检测(BCD)的对比学习增强模块

基于Contextrast论文思想，针对Change3D的perception features进行增强：
1. 多尺度embedding投影
2. 二类锚点（Change/No-change）
3. 跨尺度锚点融合（注入全局context）
4. Pixel-Anchor对比损失
5. BANE边界感知负样本采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


class MultiScaleEmbeddingHeads(nn.Module):
    """
    多尺度embedding投影头
    
    将每个尺度的perception features投影到统一的embedding空间
    """
    
    def __init__(self, in_channels: List[int], embed_dim: int = 256):
        """
        Args:
            in_channels: 各尺度输入通道数 [24, 24, 48, 96]
            embed_dim: embedding维度
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_scales = len(in_channels)
        
        # 为每个尺度创建投影头
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            )
            for in_ch in in_channels
        ])
        
    def forward(self, perception_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            perception_features: 各尺度perception features [p0, p1, p2, p3]
                每个 pi shape: [B, Ci, Hi, Wi]
        
        Returns:
            embeddings: 各尺度embedding features
                每个 vi shape: [B, D, Hi, Wi]，D=embed_dim
        """
        embeddings = []
        
        for i, (percep_feat, proj_head) in enumerate(
            zip(perception_features, self.projection_heads)
        ):
            embed = proj_head(percep_feat)
            # L2归一化（论文中提到有助于稳定训练）
            embed = F.normalize(embed, p=2, dim=1)
            embeddings.append(embed)
        
        return embeddings


class BinaryChangeAnchors(nn.Module):
    """
    二类锚点（Change / No-change）计算
    
    为每个尺度计算两个类别的representative anchors
    """
    
    def __init__(self, embed_dim: int = 256):
        """
        Args:
            embed_dim: embedding维度
        """
        super().__init__()
        self.embed_dim = embed_dim
        
    def compute_anchors(
        self, 
        embeddings: List[torch.Tensor], 
        gt_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        计算各尺度的Change/No-change锚点
        
        Args:
            embeddings: 各尺度embedding [v0, v1, v2, v3]
                每个 shape: [B, D, Hi, Wi]
            gt_mask: Ground truth变化掩码 [B, 1, H, W]，值为0(no-change)或1(change)
        
        Returns:
            anchors_change: 各尺度change类锚点，每个shape [B, D]
            anchors_nochange: 各尺度no-change类锚点，每个shape [B, D]
        """
        anchors_change = []
        anchors_nochange = []
        
        batch_size = gt_mask.shape[0]
        
        for embed in embeddings:
            _, _, H, W = embed.shape
            
            # 下采样GT mask到当前尺度
            gt_downsampled = F.interpolate(
                gt_mask.float(), 
                size=(H, W), 
                mode='nearest'
            )  # [B, 1, H, W]
            
            # 展平为 [B, D, H*W]
            embed_flat = embed.view(batch_size, self.embed_dim, -1)  # [B, D, H*W]
            gt_flat = gt_downsampled.view(batch_size, 1, -1)  # [B, 1, H*W]
            
            # 计算change类锚点（所有change像素的均值）
            change_mask = (gt_flat == 1).float()  # [B, 1, H*W]
            change_count = change_mask.sum(dim=2, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
            
            anchor_chg = (embed_flat * change_mask).sum(dim=2) / change_count.squeeze(-1)  # [B, D]
            
            # 计算no-change类锚点
            nochange_mask = (gt_flat == 0).float()  # [B, 1, H*W]
            nochange_count = nochange_mask.sum(dim=2, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
            
            anchor_nochg = (embed_flat * nochange_mask).sum(dim=2) / nochange_count.squeeze(-1)  # [B, D]
            
            # L2归一化
            anchor_chg = F.normalize(anchor_chg, p=2, dim=1)
            anchor_nochg = F.normalize(anchor_nochg, p=2, dim=1)
            
            anchors_change.append(anchor_chg)
            anchors_nochange.append(anchor_nochg)
        
        return anchors_change, anchors_nochange


class CrossScaleAnchorFusion(nn.Module):
    """
    跨尺度锚点融合
    
    用最深层anchor给浅层anchor注入全局语义参考系
    """
    
    def __init__(self, w_low: float = 0.7, w_high: float = 0.3):
        """
        Args:
            w_low: 本层anchor权重
            w_high: 深层anchor权重
        """
        super().__init__()
        self.w_low = w_low
        self.w_high = w_high
        
    def forward(
        self, 
        anchors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            anchors: 各尺度锚点 [A0, A1, A2, A3]
                每个 shape: [B, D]
        
        Returns:
            fused_anchors: 融合后的锚点，shape同输入
        """
        num_scales = len(anchors)
        deepest_anchor = anchors[-1]  # 最深层anchor
        
        fused_anchors = []
        
        for i in range(num_scales):
            if i == num_scales - 1:
                # 最深层不需要融合
                fused_anchors.append(deepest_anchor)
            else:
                # 浅层anchor与深层anchor加权融合
                fused = self.w_low * anchors[i] + self.w_high * deepest_anchor
                # 归一化
                fused = F.normalize(fused, p=2, dim=1)
                fused_anchors.append(fused)
        
        return fused_anchors


class BANESampling:
    """
    Boundary-Aware Negative (BANE) Sampling
    
    基于预测错误区域的距离变换，优先采样靠近边界的hard negatives
    """
    
    def __init__(self, sample_ratio: float = 0.5):
        """
        Args:
            sample_ratio: 采样比例（采样最靠近边界的K%像素）
        """
        self.sample_ratio = sample_ratio
        
    def compute_distance_map(
        self, 
        pred: torch.Tensor, 
        gt: torch.Tensor
    ) -> torch.Tensor:
        """
        计算预测错误区域的距离图
        
        Args:
            pred: 预测结果 [B, 1, H, W]，值为0或1
            gt: Ground truth [B, 1, H, W]，值为0或1
        
        Returns:
            distance_maps: 距离图 [B, 1, H, W]
        """
        batch_size = pred.shape[0]
        device = pred.device
        
        distance_maps = []
        
        for b in range(batch_size):
            pred_b = pred[b, 0].cpu().numpy()  # [H, W]
            gt_b = gt[b, 0].cpu().numpy()  # [H, W]
            
            # 计算错误区域：预测!=GT
            error_mask = (pred_b != gt_b).astype(np.uint8)  # [H, W]
            
            if error_mask.sum() == 0:
                # 没有错误，距离图全为0
                dist_map = np.zeros_like(error_mask, dtype=np.float32)
            else:
                # 距离变换：计算每个错误像素到边界的距离
                dist_map = distance_transform_edt(error_mask).astype(np.float32)
            
            distance_maps.append(dist_map)
        
        # 转换回tensor
        distance_maps = np.stack(distance_maps, axis=0)  # [B, H, W]
        distance_maps = torch.from_numpy(distance_maps).unsqueeze(1).to(device)  # [B, 1, H, W]
        
        return distance_maps
    
    def sample_hard_negatives(
        self,
        embeddings: torch.Tensor,
        pred: torch.Tensor,
        gt: torch.Tensor,
        target_class: int,
        max_samples: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样hard negative samples（内存优化版）
        
        Args:
            embeddings: embedding特征 [B, D, H, W]
            pred: 预测结果 [B, 1, H, W]
            gt: Ground truth [B, 1, H, W]
            target_class: 目标类别（0: no-change, 1: change）
            max_samples: 每个batch最多采样的负样本数（内存优化）
        
        Returns:
            neg_samples: 负样本embeddings [B, K, D]，K<=max_samples
            sample_mask: 采样掩码 [B, 1, H, W]
        """
        batch_size, embed_dim, H, W = embeddings.shape
        device = embeddings.device
        
        # 计算距离图
        distance_maps = self.compute_distance_map(pred, gt)  # [B, 1, H, W]
        
        # 找出错误预测区域（且GT=target_class）
        error_mask = ((pred != gt) & (gt == target_class)).float()  # [B, 1, H, W]
        
        neg_samples_list = []
        sample_masks = []
        
        for b in range(batch_size):
            error_b = error_mask[b, 0]  # [H, W]
            dist_b = distance_maps[b, 0]  # [H, W]
            embed_b = embeddings[b]  # [D, H, W]
            
            # 获取错误像素的位置
            error_coords = torch.nonzero(error_b, as_tuple=False)  # [N, 2]
            
            if error_coords.shape[0] == 0:
                # 没有错误像素，随机采样少量样本
                num_samples = min(max_samples // 4, 64)  # 大幅减少随机采样数量
                rand_idx = torch.randperm(H * W, device=device)[:num_samples]
                h_idx = rand_idx // W
                w_idx = rand_idx % W
                
                neg_sample_b = embed_b[:, h_idx, w_idx].t()  # [num_samples, D]
                
                # 创建采样掩码
                sample_mask_b = torch.zeros(H, W, device=device)
                sample_mask_b[h_idx, w_idx] = 1.0
            else:
                # 根据距离排序，选择最靠近边界的K%
                error_h = error_coords[:, 0]
                error_w = error_coords[:, 1]
                
                distances = dist_b[error_h, error_w]  # [N]
                
                # 计算采样数量（限制最大值）
                num_candidates = len(distances)
                num_samples = max(1, int(num_candidates * self.sample_ratio))
                num_samples = min(num_samples, max_samples)  # 强制限制最大采样数
                
                # 排序（距离小的优先）
                _, sorted_idx = torch.topk(
                    distances, 
                    k=min(num_samples, num_candidates), 
                    largest=False
                )
                
                selected_h = error_h[sorted_idx]
                selected_w = error_w[sorted_idx]
                
                neg_sample_b = embed_b[:, selected_h, selected_w].t()  # [K, D]
                
                # 创建采样掩码
                sample_mask_b = torch.zeros(H, W, device=device)
                sample_mask_b[selected_h, selected_w] = 1.0
            
            neg_samples_list.append(neg_sample_b)
            sample_masks.append(sample_mask_b)
        
        # 统一batch中的样本数（padding到最大样本数）
        actual_max_samples = max(ns.shape[0] for ns in neg_samples_list)
        actual_max_samples = min(actual_max_samples, max_samples)  # 再次确保不超限
        
        padded_neg_samples = []
        for ns in neg_samples_list:
            current_samples = ns.shape[0]
            if current_samples > actual_max_samples:
                # 如果超过限制，随机采样
                rand_idx = torch.randperm(current_samples)[:actual_max_samples]
                ns = ns[rand_idx]
            elif current_samples < actual_max_samples:
                # 如果不足，padding
                padding = torch.zeros(
                    actual_max_samples - current_samples, 
                    embed_dim, 
                    device=device
                )
                ns = torch.cat([ns, padding], dim=0)
            padded_neg_samples.append(ns)
        
        neg_samples = torch.stack(padded_neg_samples, dim=0)  # [B, K, D]
        sample_masks = torch.stack(sample_masks, dim=0).unsqueeze(1)  # [B, 1, H, W]
        
        return neg_samples, sample_masks


class PixelAnchorContrastiveLoss(nn.Module):
    """
    Pixel-Anchor对比损失（内存优化版）
    
    让每个像素embedding拉近其GT类锚点，远离另一类锚点
    使用分块计算避免大tensor
    """
    
    def __init__(self, temperature: float = 0.07, chunk_size: int = 4096):
        """
        Args:
            temperature: 温度参数
            chunk_size: 分块大小，减少内存使用
        """
        super().__init__()
        self.temperature = temperature
        self.chunk_size = chunk_size
        
    def forward(
        self,
        embeddings: torch.Tensor,
        anchor_pos: torch.Tensor,
        anchor_neg: torch.Tensor,
        gt_mask: torch.Tensor,
        neg_samples: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算Pixel-Anchor对比损失（内存优化）
        
        Args:
            embeddings: embedding特征 [B, D, H, W]
            anchor_pos: 正类锚点 [B, D]
            anchor_neg: 负类锚点 [B, D]
            gt_mask: Ground truth [B, 1, H, W]
            neg_samples: BANE采样的负样本 [B, K, D]（可选）
        
        Returns:
            loss: 对比损失标量
        """
        batch_size, embed_dim, H, W = embeddings.shape
        num_pixels = H * W
        
        # 展平embeddings
        embed_flat = embeddings.view(batch_size, embed_dim, -1)  # [B, D, H*W]
        embed_flat = embed_flat.permute(0, 2, 1)  # [B, H*W, D]
        
        # 展平GT
        gt_flat = gt_mask.view(batch_size, 1, -1).squeeze(1)  # [B, H*W]
        
        # 扩展anchor维度
        anchor_pos = anchor_pos.unsqueeze(1)  # [B, 1, D]
        anchor_neg = anchor_neg.unsqueeze(1)  # [B, 1, D]
        
        # === 内存优化：分块计算 ===
        total_loss = 0.0
        total_valid = 0
        
        # 计算需要多少个chunk
        num_chunks = (num_pixels + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, num_pixels)
            
            # 当前chunk的embeddings
            embed_chunk = embed_flat[:, start_idx:end_idx, :]  # [B, chunk_size, D]
            gt_chunk = gt_flat[:, start_idx:end_idx]  # [B, chunk_size]
            
            # 计算与正类anchor的相似度
            sim_pos = torch.bmm(embed_chunk, anchor_pos.transpose(1, 2))  # [B, chunk_size, 1]
            sim_pos = sim_pos.squeeze(-1) / self.temperature  # [B, chunk_size]
            
            # 计算与负类anchor的相似度
            sim_neg_anchor = torch.bmm(embed_chunk, anchor_neg.transpose(1, 2))  # [B, chunk_size, 1]
            sim_neg_anchor = sim_neg_anchor.squeeze(-1) / self.temperature  # [B, chunk_size]
            
            # 如果有BANE采样的负样本
            if neg_samples is not None and neg_samples.shape[1] > 0:
                # 只使用前K个最重要的负样本（进一步减少内存）
                K = min(neg_samples.shape[1], 256)  # 限制最多256个负样本
                neg_samples_reduced = neg_samples[:, :K, :]  # [B, K, D]
                
                # 分块计算与负样本的相似度
                sim_neg_samples = torch.bmm(
                    embed_chunk, 
                    neg_samples_reduced.transpose(1, 2)
                )  # [B, chunk_size, K]
                sim_neg_samples = sim_neg_samples / self.temperature
                
                # 合并所有负样本相似度
                sim_neg = torch.cat([
                    sim_neg_anchor.unsqueeze(-1),  # [B, chunk_size, 1]
                    sim_neg_samples  # [B, chunk_size, K]
                ], dim=-1)  # [B, chunk_size, 1+K]
            else:
                sim_neg = sim_neg_anchor.unsqueeze(-1)  # [B, chunk_size, 1]
            
            # InfoNCE损失
            exp_sim_pos = torch.exp(sim_pos)  # [B, chunk_size]
            exp_sim_neg = torch.exp(sim_neg).sum(dim=-1)  # [B, chunk_size]
            
            loss_chunk = -torch.log(exp_sim_pos / (exp_sim_pos + exp_sim_neg + 1e-8))
            
            # 只计算有效像素的损失
            valid_mask = (gt_chunk >= 0).float()  # [B, chunk_size]
            loss_chunk = (loss_chunk * valid_mask).sum()
            
            total_loss += loss_chunk
            total_valid += valid_mask.sum()
        
        # 归一化
        final_loss = total_loss / (total_valid + 1e-8)
        
        return final_loss


class BCDContrastiveModule(nn.Module):
    """
    BCD对比学习完整模块（仅训练时使用）
    
    整合：
    1. 多尺度embedding
    2. 二类锚点计算
    3. 跨尺度锚点融合
    4. BANE采样
    5. Pixel-Anchor对比损失
    """
    
    def __init__(
        self,
        in_channels: List[int] = [24, 24, 48, 96],
        embed_dim: int = 256,
        temperature: float = 0.07,
        sample_ratio: float = 0.5,
        w_low: float = 0.7,
        w_high: float = 0.3,
        scale_weights: Optional[List[float]] = None,
        max_neg_samples: int = 256,  # 新增：最大负样本数
        loss_chunk_size: int = 4096  # 新增：损失计算分块大小
    ):
        """
        Args:
            in_channels: 各尺度输入通道数
            embed_dim: embedding维度
            temperature: 对比学习温度
            sample_ratio: BANE采样比例
            w_low: 本层anchor权重
            w_high: 深层anchor权重
            scale_weights: 各尺度损失权重
            max_neg_samples: 最大负样本数（内存优化）
            loss_chunk_size: 损失计算分块大小（内存优化）
        """
        super().__init__()
        
        self.num_scales = len(in_channels)
        
        # 默认尺度权重（深层权重更大）
        if scale_weights is None:
            scale_weights = [0.1, 0.4, 0.7, 1.0]
        self.scale_weights = scale_weights
        
        # 内存优化参数
        self.max_neg_samples = max_neg_samples
        
        # 1. 多尺度embedding投影头
        self.embedding_heads = MultiScaleEmbeddingHeads(in_channels, embed_dim)
        
        # 2. 二类锚点计算
        self.anchor_computer = BinaryChangeAnchors(embed_dim)
        
        # 3. 跨尺度锚点融合
        self.anchor_fusion = CrossScaleAnchorFusion(w_low, w_high)
        
        # 4. BANE采样
        self.bane_sampling = BANESampling(sample_ratio)
        
        # 5. Pixel-Anchor对比损失
        self.pa_loss = PixelAnchorContrastiveLoss(temperature, chunk_size=loss_chunk_size)
        
    def forward(
        self,
        perception_features: List[torch.Tensor],
        gt_mask: torch.Tensor,
        pred_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播（仅训练时调用）
        
        Args:
            perception_features: 各尺度perception features [p0, p1, p2, p3]
            gt_mask: Ground truth变化掩码 [B, 1, H, W]
            pred_mask: 预测变化掩码 [B, 1, H, W]
        
        Returns:
            total_loss: 总对比损失
            loss_dict: 各尺度损失详情
        """
        # 1. 投影到embedding空间
        embeddings = self.embedding_heads(perception_features)
        
        # 2. 计算各尺度的二类锚点
        anchors_change, anchors_nochange = self.anchor_computer.compute_anchors(
            embeddings, gt_mask
        )
        
        # 3. 跨尺度锚点融合
        fused_anchors_change = self.anchor_fusion(anchors_change)
        fused_anchors_nochange = self.anchor_fusion(anchors_nochange)
        
        # 4. 计算各尺度的对比损失
        total_loss = 0.0
        loss_dict = {}
        
        for i in range(self.num_scales):
            embed = embeddings[i]
            anchor_chg = fused_anchors_change[i]
            anchor_nochg = fused_anchors_nochange[i]
            
            _, _, H, W = embed.shape
            
            # 下采样GT和pred到当前尺度
            gt_down = F.interpolate(gt_mask.float(), size=(H, W), mode='nearest')
            pred_down = F.interpolate(pred_mask.float(), size=(H, W), mode='nearest')
            
            # 对change类：正类是change，负类是no-change
            # BANE采样：采样GT=1但pred错误的hard negatives
            neg_samples_chg, _ = self.bane_sampling.sample_hard_negatives(
                embed, pred_down, gt_down, target_class=1, max_samples=self.max_neg_samples
            )
            
            loss_chg = self.pa_loss(
                embed, anchor_chg, anchor_nochg, gt_down, neg_samples_chg
            )
            
            # 对no-change类：正类是no-change，负类是change
            neg_samples_nochg, _ = self.bane_sampling.sample_hard_negatives(
                embed, pred_down, gt_down, target_class=0, max_samples=self.max_neg_samples
            )
            
            loss_nochg = self.pa_loss(
                embed, anchor_nochg, anchor_chg, 1 - gt_down, neg_samples_nochg
            )
            
            # 尺度损失（两类损失平均）
            scale_loss = (loss_chg + loss_nochg) / 2.0
            
            # 加权累加
            weighted_loss = self.scale_weights[i] * scale_loss
            total_loss += weighted_loss
            
            loss_dict[f'scale_{i}_loss'] = scale_loss.item()
            loss_dict[f'scale_{i}_change_loss'] = loss_chg.item()
            loss_dict[f'scale_{i}_nochange_loss'] = loss_nochg.item()
        
        # 归一化（除以尺度权重和）
        total_loss = total_loss / sum(self.scale_weights)
        loss_dict['total_contrastive_loss'] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================================
# 使用示例和测试代码
# ============================================================

if __name__ == '__main__':
    """测试BCD对比学习模块"""
    
    # 模拟数据
    batch_size = 4
    in_channels = [24, 24, 48, 96]
    
    # 模拟多尺度perception features
    perception_features = [
        torch.randn(batch_size, in_channels[0], 64, 64),  # p0
        torch.randn(batch_size, in_channels[1], 32, 32),  # p1
        torch.randn(batch_size, in_channels[2], 16, 16),  # p2
        torch.randn(batch_size, in_channels[3], 8, 8),    # p3
    ]
    
    # 模拟GT和预测
    gt_mask = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    pred_mask = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # 创建对比学习模块
    contrast_module = BCDContrastiveModule(
        in_channels=in_channels,
        embed_dim=256,
        temperature=0.07,
        sample_ratio=0.5
    )
    
    # 前向传播
    total_loss, loss_dict = contrast_module(
        perception_features, gt_mask, pred_mask
    )
    
    print(f"Total Contrastive Loss: {total_loss.item():.4f}")
    print("\nDetailed Loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试梯度反传
    total_loss.backward()
    print("\n✅ Backward pass successful!")
    
    # 打印模块参数量
    total_params = sum(p.numel() for p in contrast_module.parameters())
    trainable_params = sum(p.numel() for p in contrast_module.parameters() if p.requires_grad)
    print(f"\nModule Parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")