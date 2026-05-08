# Copyright (c) Unified Trainer: Differential Attention + BCD Contrastive
# All rights reserved.

"""
统一训练器: 整合差分注意力和BCD对比学习

架构:
    Input (Pre + Post)
        ↓
    UnifiedEncoder (差分注意力增强)
        ↓
    ┌──────────┴──────────┐
    ↓                     ↓
  Task Branch      Contrastive Branch
  (Decoder)         (BCD Module)
    ↓                     ↓
  L_task            L_contrast
    └──────────┬──────────┘
               ↓
      L_total = L_task + α*L_contrast
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any

# 导入编码器和解码器
from unified_encoder import UnifiedEncoder  # 上面创建的
from model.change_decoder import ChangeDecoder  # 项目中已有
from bcd_contrastive_module import BCDContrastiveModule  # 之前创建的
from scipy.ndimage import distance_transform_edt


class UnifiedTrainer(nn.Module):
    """
    统一训练器
    
    整合功能:
    1. 差分注意力增强感知特征
    2. 跨层级注意力传播 (可选)
    3. BCD对比学习优化特征表示
    """
    
    def __init__(self, args: Any):
        """
        Args:
            args: 配置参数，需要包含:
                - pretrained: X3D预训练权重路径
                - num_perception_frame: 感知帧数量
                - in_height, in_width: 输入尺寸
                - num_class: 类别数（BCD为1）
                - embed_dim: 对比学习embedding维度
                - contrast_weight: 对比损失权重
                - use_differential_attention: 是否使用差分注意力
                - use_cross_level: 是否使用跨层级传播
                - ... (其他差分注意力配置)
        """
        super().__init__()
        self.args = args
        
        # 嵌入维度配置
        self.embed_dims = [24, 24, 48, 96]  # X3D各层输出通道
        
        # ========== 编码器: 差分注意力增强 ==========
        from unified_encoder import UnifiedEncoder
        
        self.encoder = UnifiedEncoder(
            args,
            embed_dims=self.embed_dims,
            use_differential_attention=getattr(args, 'use_differential_attention', False),
            use_gating=getattr(args, 'use_gating', False),
            use_channel_attention=getattr(args, 'use_channel_attention', False),
            use_spatial_attention=getattr(args, 'use_spatial_attention', False),
            use_cross_level=getattr(args, 'use_cross_level', False),
            use_top_down=getattr(args, 'use_top_down', False),
            use_bottom_up=getattr(args, 'use_bottom_up', False),
        )
        
        # ========== 任务分支: Change Decoder ==========
        from model.change_decoder import ChangeDecoder
        from model.utils import weight_init
        
        self.decoder = ChangeDecoder(
            args, 
            in_dim=self.embed_dims, 
            has_sigmoid=True
        )
        weight_init(self.decoder)
        
        # ========== 对比学习分支: BCD Contrastive ==========
        # 注意: 这个模块需要从之前生成的文件中导入
        # 这里我们需要复制必要的类
        
        # 由于导入问题，我们在这里内联必要的BCD模块
        # 实际使用时应该从单独的文件导入
        self._init_contrastive_module(args)
        
        # ========== 损失权重 ==========
        self.contrast_weight = getattr(args, 'contrast_weight', 0.1)
        
        print(f"\n{'='*60}")
        print(f"Unified Trainer Initialized")
        print(f"{'='*60}")
        print(f"  Contrast Weight: {self.contrast_weight}")
        print(f"  Embed Dim: {getattr(args, 'embed_dim', 256)}")
        print(f"  Temperature: {getattr(args, 'temperature', 0.07)}")
        print(f"{'='*60}\n")
    
    def _init_contrastive_module(self, args):
        """初始化对比学习模块（内联版本）"""
        
        embed_dim = getattr(args, 'embed_dim', 256)
        temperature = getattr(args, 'temperature', 0.07)
        
        # 多尺度Embedding投影头
        class MultiScaleEmbeddingHeads(nn.Module):
            def __init__(self, in_channels, embed_dim):
                super().__init__()
                self.projection_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(embed_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
                    )
                    for in_ch in in_channels
                ])
            
            def forward(self, features):
                embeddings = []
                for feat, proj in zip(features, self.projection_heads):
                    embed = proj(feat)
                    embed = F.normalize(embed, p=2, dim=1)
                    embeddings.append(embed)
                return embeddings
        
        self.embedding_heads = MultiScaleEmbeddingHeads(self.embed_dims, embed_dim)
        
        # 对比学习参数
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.scale_weights = [0.1, 0.4, 0.7, 1.0]  # 深层权重更大
        self.max_neg_samples = getattr(args, 'max_neg_samples', 256)
        self.chunk_size = getattr(args, 'chunk_size', 4096)
        
        print(f"✅ Initialized BCD Contrastive Module")
        print(f"   Embedding Heads: {len(self.embedding_heads.projection_heads)}")
        print(f"   Scale Weights: {self.scale_weights}")
    
    def compute_anchors(self, embeddings: List[torch.Tensor], gt_mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        计算二类锚点（Change / No-change）- 修复版本
        
        Args:
            embeddings: 各尺度embedding [v0, v1, v2, v3]
            gt_mask: 真值mask [B, 1, H, W]
        
        Returns:
            anchors_change: 各尺度change锚点 [a0_c, ..., a3_c], 每个 [B, D]
            anchors_nochange: 各尺度no-change锚点 [a0_nc, ..., a3_nc]
        """
        anchors_change = []
        anchors_nochange = []
        
        for embed in embeddings:
            B, D, H, W = embed.shape
            
            # 下采样GT到当前尺度
            gt_resized = F.interpolate(
                gt_mask.float(), 
                size=(H, W), 
                mode='nearest'
            ).long()
            
            # 展平
            embed_flat = embed.view(B, D, -1).permute(0, 2, 1)  # [B, H*W, D]
            gt_flat = gt_resized.view(B, -1)  # [B, H*W]
            
            # 计算每个batch的锚点
            batch_anchors_c = []
            batch_anchors_nc = []
            
            for b in range(B):
                # 提取两类像素
                change_mask = gt_flat[b] == 1
                nochange_mask = gt_flat[b] == 0
                
                change_embed = embed_flat[b][change_mask]  # [N_c, D]
                nochange_embed = embed_flat[b][nochange_mask]  # [N_nc, D]
                
                # 计算均值作为锚点 - 关键修复：确保至少有一个像素
                if len(change_embed) > 0:
                    anchor_c = change_embed.mean(dim=0)  # [D]
                    anchor_c = F.normalize(anchor_c, p=2, dim=0)
                else:
                    # 如果没有change像素，使用随机初始化的归一化向量
                    anchor_c = torch.randn(D, device=embed.device)
                    anchor_c = F.normalize(anchor_c, p=2, dim=0)
                
                if len(nochange_embed) > 0:
                    anchor_nc = nochange_embed.mean(dim=0)  # [D]
                    anchor_nc = F.normalize(anchor_nc, p=2, dim=0)
                else:
                    # 如果没有no-change像素，使用随机初始化的归一化向量
                    anchor_nc = torch.randn(D, device=embed.device)
                    anchor_nc = F.normalize(anchor_nc, p=2, dim=0)
                
                batch_anchors_c.append(anchor_c)
                batch_anchors_nc.append(anchor_nc)
            
            anchors_change.append(torch.stack(batch_anchors_c))  # [B, D]
            anchors_nochange.append(torch.stack(batch_anchors_nc))
        
        return anchors_change, anchors_nochange
    
    def fuse_anchors_cross_scale(
        self, 
        anchors_change: List[torch.Tensor],
        anchors_nochange: List[torch.Tensor],
        fusion_weights: List[float] = [0.3, 0.3, 0.3, 0.0]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        跨尺度锚点融合（注入全局语义）
        
        Args:
            anchors_change: 各尺度change锚点
            anchors_nochange: 各尺度no-change锚点
            fusion_weights: 全局锚点的融合权重（最深层为全局锚点）
        
        Returns:
            fused_anchors_change: 融合后的change锚点
            fused_anchors_nochange: 融合后的no-change锚点
        """
        # 全局锚点 = 最深层锚点
        global_anchor_c = anchors_change[-1]  # [B, D]
        global_anchor_nc = anchors_nochange[-1]
        
        fused_anchors_c = []
        fused_anchors_nc = []
        
        for i in range(len(anchors_change)):
            local_c = anchors_change[i]
            local_nc = anchors_nochange[i]
            
            # 融合: (1-α)*local + α*global
            alpha = fusion_weights[i]
            fused_c = (1 - alpha) * local_c + alpha * global_anchor_c
            fused_nc = (1 - alpha) * local_nc + alpha * global_anchor_nc
            
            # 重新归一化
            fused_c = F.normalize(fused_c, p=2, dim=1)
            fused_nc = F.normalize(fused_nc, p=2, dim=1)
            
            fused_anchors_c.append(fused_c)
            fused_anchors_nc.append(fused_nc)
        
        return fused_anchors_c, fused_anchors_nc
    
    def sample_hard_negatives_bane(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        embeddings: torch.Tensor,
        max_samples: int = 256
    ) -> Optional[torch.Tensor]:
        """
        BANE: 边界感知负样本采样 - 修复版本
        
        Args:
            pred: 预测mask [B, 1, H, W]
            gt: 真值mask [B, 1, H, W]
            embeddings: embedding特征 [B, D, H, W]
            max_samples: 最大负样本数
        
        Returns:
            neg_samples: [B, K, D] 或 None（保证每个batch都有对应项）
        """
        B, D, H, W = embeddings.shape
        
        # 下采样pred和gt到embedding尺寸
        pred_resized = F.interpolate(pred.float(), size=(H, W), mode='nearest')
        gt_resized = F.interpolate(gt.float(), size=(H, W), mode='nearest')
        
        # 转为二值
        pred_binary = (pred_resized > 0.5).long().squeeze(1)  # [B, H, W]
        gt_binary = gt_resized.long().squeeze(1)
        
        # 为每个batch创建负样本（包括空的）- 关键修复！
        batch_negatives = []
        
        for b in range(B):
            # 错误区域
            error_mask = (pred_binary[b] != gt_binary[b]).cpu().numpy()
            
            # 如果错误太少，使用零填充（而不是跳过）
            if error_mask.sum() < 10:
                batch_negatives.append(torch.zeros(1, D, device=embeddings.device))
                continue
            
            try:
                # 距离变换（边界距离）
                dist_map = distance_transform_edt(error_mask)
                dist_tensor = torch.from_numpy(dist_map).to(embeddings.device)
                
                # 提取错误区域的坐标
                error_coords = error_mask.nonzero()
                
                if len(error_coords[0]) == 0:
                    batch_negatives.append(torch.zeros(1, D, device=embeddings.device))
                    continue
                
                # 提取对应的距离值
                error_dists = dist_tensor[error_coords[0], error_coords[1]]
                
                # 采样距离最小的K个（靠近边界）
                K = min(len(error_coords[0]), max_samples)
                _, topk_idx = torch.topk(error_dists, K, largest=False)
                
                # 提取对应的embedding
                sampled_coords = (error_coords[0][topk_idx.cpu()], 
                                error_coords[1][topk_idx.cpu()])
                
                neg_embed = embeddings[b, :, sampled_coords[0], sampled_coords[1]]  # [D, K]
                neg_embed = neg_embed.permute(1, 0)  # [K, D]
                
                batch_negatives.append(neg_embed)
            
            except Exception as e:
                # 如果采样失败，使用零填充
                batch_negatives.append(torch.zeros(1, D, device=embeddings.device))
        
        # 检查是否所有batch都是零向量
        if all(n.shape[0] == 1 and torch.norm(n) < 0.1 for n in batch_negatives):
            return None
        
        # Padding到相同长度
        max_len = max(n.shape[0] for n in batch_negatives)
        padded_negatives = []
        
        for neg in batch_negatives:
            if neg.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - neg.shape[0], D, 
                    device=neg.device, dtype=neg.dtype
                )
                neg = torch.cat([neg, padding], dim=0)
            padded_negatives.append(neg)
        
        # Stack成 [B, K, D] - 确保batch维度 = B
        return torch.stack(padded_negatives, dim=0)
    
    def pixel_anchor_contrastive_loss(
        self,
        embeddings: List[torch.Tensor],
        anchors_change: List[torch.Tensor],
        anchors_nochange: List[torch.Tensor],
        gt_mask: torch.Tensor,
        neg_samples_list: List[Optional[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        多尺度Pixel-Anchor对比损失 - 修复版本
        
        Args:
            embeddings: 各尺度embedding
            anchors_change: 各尺度change锚点
            anchors_nochange: 各尺度no-change锚点
            gt_mask: 真值mask
            neg_samples_list: 各尺度负样本
        
        Returns:
            total_loss: 加权多尺度损失
        """
        total_loss = 0.0
        valid_scales = 0
        
        for scale_idx, (embed, anchor_c, anchor_nc, weight) in enumerate(
            zip(embeddings, anchors_change, anchors_nochange, self.scale_weights)
        ):
            B, D, H, W = embed.shape
            
            # 下采样GT
            gt_resized = F.interpolate(
                gt_mask.float(), size=(H, W), mode='nearest'
            ).long()
            
            # 展平
            embed_flat = embed.view(B, D, -1).permute(0, 2, 1)  # [B, HW, D]
            gt_flat = gt_resized.view(B, -1)  # [B, HW]
            
            # 分块计算避免OOM
            scale_loss = 0.0
            total_pixels = 0
            
            for b in range(B):
                embed_b = embed_flat[b]  # [HW, D]
                gt_b = gt_flat[b]  # [HW]
                anchor_c_b = anchor_c[b]  # [D]
                anchor_nc_b = anchor_nc[b]  # [D]
                
                # 检查是否有足够的像素
                num_change = (gt_b == 1).sum().item()
                num_nochange = (gt_b == 0).sum().item()
                
                if num_change == 0 or num_nochange == 0:
                    # 跳过这个batch（缺少某个类别）
                    continue
                
                # 确定正锚点 - 修复：使用expand而不是where
                is_change = gt_b == 1  # [HW]
                
                # 为每个像素选择对应的正负锚点
                anchor_pos = torch.where(
                    is_change.unsqueeze(1).expand(-1, D),  # [HW, D]
                    anchor_c_b.unsqueeze(0).expand(len(gt_b), -1),  # [HW, D]
                    anchor_nc_b.unsqueeze(0).expand(len(gt_b), -1)   # [HW, D]
                )  # [HW, D]
                
                anchor_neg = torch.where(
                    is_change.unsqueeze(1).expand(-1, D),
                    anchor_nc_b.unsqueeze(0).expand(len(gt_b), -1),
                    anchor_c_b.unsqueeze(0).expand(len(gt_b), -1)
                )  # [HW, D]
                
                # 分块计算
                num_pixels = embed_b.shape[0]
                
                for start in range(0, num_pixels, self.chunk_size):
                    end = min(start + self.chunk_size, num_pixels)
                    chunk_embed = embed_b[start:end]  # [chunk, D]
                    chunk_pos = anchor_pos[start:end]  # [chunk, D]
                    chunk_neg_anchor = anchor_neg[start:end]  # [chunk, D]
                    
                    # 正样本相似度
                    sim_pos = (chunk_embed * chunk_pos).sum(dim=1) / self.temperature  # [chunk]
                    
                    # 负样本相似度（锚点）
                    sim_neg_anchor = (chunk_embed * chunk_neg_anchor).sum(dim=1) / self.temperature
                    
                    # 添加BANE负样本
                    if neg_samples_list and neg_samples_list[scale_idx] is not None:
                        neg_samples = neg_samples_list[scale_idx][b]  # [K, D]
                        # 过滤掉padding的零向量
                        neg_norm = torch.norm(neg_samples, dim=1)
                        valid_neg = neg_samples[neg_norm > 0.1]  # [K', D]
                        
                        if len(valid_neg) > 0:
                            sim_neg_hard = (chunk_embed @ valid_neg.T) / self.temperature  # [chunk, K']
                            sim_neg = torch.cat([
                                sim_neg_anchor.unsqueeze(1), 
                                sim_neg_hard
                            ], dim=1)
                        else:
                            sim_neg = sim_neg_anchor.unsqueeze(1)
                    else:
                        sim_neg = sim_neg_anchor.unsqueeze(1)
                    
                    # InfoNCE损失
                    exp_pos = torch.exp(sim_pos)
                    exp_neg = torch.exp(sim_neg).sum(dim=1)
                    
                    loss_chunk = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8))
                    
                    scale_loss += loss_chunk.sum()
                    total_pixels += len(chunk_embed)
            
            # 如果这个尺度有有效像素，添加到总损失
            if total_pixels > 0:
                scale_loss = scale_loss / total_pixels
                total_loss += weight * scale_loss
                valid_scales += 1
        
        # 如果没有有效尺度，返回0损失
        if valid_scales == 0:
            return torch.tensor(0.0, device=embeddings[0].device)
        
        # 归一化权重
        total_loss = total_loss / sum(self.scale_weights[:valid_scales])
        
        return total_loss
    
    def forward(
        self, 
        pre_img: torch.Tensor, 
        post_img: torch.Tensor,
        gt_mask: torch.Tensor = None,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        统一前向传播
        
        Args:
            pre_img: 前时相 [B, 3, H, W]
            post_img: 后时相 [B, 3, H, W]
            gt_mask: 真值mask [B, 1, H, W] (训练时需要)
            return_features: 是否返回中间特征
        
        Returns:
            pred: 变化预测 [B, 1, H, W]
            contrast_loss: 对比损失 (训练时)
            features_dict: 中间特征字典 (if return_features)
        """
        # ========== 编码器 ==========
        final_features, pre_post_features = self.encoder(pre_img, post_img)
        
        # 提取perception features (每个尺度的第一个)
        perception_feats = [layer[0] for layer in final_features]
        
        # ========== 任务分支: 解码器 ==========
        pred = self.decoder(perception_feats)
        
        # ========== 对比学习分支 ==========
        contrast_loss = None
        features_dict = None
        
        if self.training and gt_mask is not None:
            # 1. Embedding投影
            embeddings = self.embedding_heads(perception_feats)
            
            # 2. 计算锚点
            anchors_change, anchors_nochange = self.compute_anchors(embeddings, gt_mask)
            
            # 3. 跨尺度锚点融合
            fused_anchors_c, fused_anchors_nc = self.fuse_anchors_cross_scale(
                anchors_change, anchors_nochange
            )
            
            # 4. BANE负样本采样
            neg_samples_list = []
            for i, embed in enumerate(embeddings):
                neg_samples = self.sample_hard_negatives_bane(
                    pred.detach(), gt_mask, embed, self.max_neg_samples
                )
                neg_samples_list.append(neg_samples)
            
            # 5. 对比损失
            contrast_loss = self.pixel_anchor_contrastive_loss(
                embeddings,
                fused_anchors_c,
                fused_anchors_nc,
                gt_mask,
                neg_samples_list
            )
            
            if return_features:
                features_dict = {
                    'embeddings': embeddings,
                    'anchors_change': fused_anchors_c,
                    'anchors_nochange': fused_anchors_nc,
                    'neg_samples': neg_samples_list,
                    'attention_maps': self.encoder.get_attention_maps()
                }
        
        return pred, contrast_loss, features_dict
    
    def get_total_loss(
        self, 
        task_loss: torch.Tensor, 
        contrast_loss: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算总损失
        
        Args:
            task_loss: 任务损失 (BCE+Dice)
            contrast_loss: 对比损失
        
        Returns:
            total_loss: 总损失
            contrast_loss_weighted: 加权后的对比损失
        """
        if contrast_loss is None:
            return task_loss, torch.tensor(0.0, device=task_loss.device)
        
        contrast_loss_weighted = self.contrast_weight * contrast_loss
        total_loss = task_loss + contrast_loss_weighted
        
        return total_loss, contrast_loss_weighted


# ========== 辅助函数 ==========

def count_parameters(model):
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: UnifiedTrainer):
    """打印模型摘要"""
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"Unified Trainer Summary")
    print(f"{'='*60}")
    print(f"Total Parameters: {total/1e6:.2f}M")
    print(f"Trainable Parameters: {trainable/1e6:.2f}M")
    print(f"\nModule Breakdown:")
    print(f"  Encoder: {count_parameters(model.encoder)[0]/1e6:.2f}M")
    print(f"  Decoder: {count_parameters(model.decoder)[0]/1e3:.1f}K")
    print(f"  Embedding Heads: {count_parameters(model.embedding_heads)[0]/1e3:.1f}K")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    """测试统一训练器"""
    from argparse import Namespace
    
    args = Namespace(
        pretrained='path/to/X3D_L.pyth',
        num_perception_frame=1,
        in_height=256,
        in_width=256,
        num_class=1,
        embed_dim=256,
        temperature=0.07,
        contrast_weight=0.1,
        max_neg_samples=256,
        chunk_size=4096,
        use_differential_attention=True,
        use_gating=True,
        use_channel_attention=True,
        use_spatial_attention=True,
        use_cross_level=True,
        use_top_down=True,
        use_bottom_up=True,
    )
    
    # 创建模型
    model = UnifiedTrainer(args).cuda()
    model.train()
    
    # 打印摘要
    print_model_summary(model)
    
    # 测试前向传播
    B = 2
    pre_img = torch.randn(B, 3, 256, 256).cuda()
    post_img = torch.randn(B, 3, 256, 256).cuda()
    gt_mask = torch.randint(0, 2, (B, 1, 256, 256)).cuda()
    
    # 训练模式
    pred, contrast_loss, features = model(
        pre_img, post_img, gt_mask, return_features=True
    )
    
    print(f"Forward Pass Test (Training):")
    print(f"  Prediction: {pred.shape}, range [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"  Contrast Loss: {contrast_loss.item():.4f}")
    
    # 任务损失
    from model.utils import BCEDiceLoss
    task_loss = BCEDiceLoss(pred, gt_mask.float())
    
    # 总损失
    total_loss, contrast_weighted = model.get_total_loss(task_loss, contrast_loss)
    
    print(f"\nLoss Computation:")
    print(f"  Task Loss: {task_loss.item():.4f}")
    print(f"  Contrast Loss: {contrast_loss.item():.4f}")
    print(f"  Contrast Weighted: {contrast_weighted.item():.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")
    
    # 推理模式
    model.eval()
    with torch.no_grad():
        pred_infer, _, _ = model(pre_img, post_img)
    
    print(f"\nForward Pass Test (Inference):")
    print(f"  Prediction: {pred_infer.shape}")
    print(f"  Contrast Loss: None (disabled in eval mode)")
    
    print(f"\n✅ Unified Trainer test passed!")