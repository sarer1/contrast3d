# Copyright (c) Unified Framework: Differential Attention + BCD Contrastive
# All rights reserved.

"""
统一框架编码器: 差分注意力 + BCD对比学习

架构流程:
1. X3D Backbone → 基础特征
2. Differential Attention → 增强感知特征  
3. Cross-Level Propagation → 多尺度融合 (可选)
4. 双分支输出:
   - Task Branch: 用于变化检测解码
   - Contrastive Branch: 用于对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from einops import repeat

from model.x3d import create_x3d


class DifferentialAttentionModule(nn.Module):
    """
    差分注意力增强模块
    
    功能: 利用前后时相差异增强感知特征
    输入: pre_feat, post_feat, percep_feat
    输出: enhanced_percep_feat
    """
    
    def __init__(self, 
                 in_channels: int, 
                 reduction: int = 16,
                 use_gating: bool = True,
                 use_channel_attention: bool = True,
                 use_spatial_attention: bool = True):
        super().__init__()
        
        self.use_gating = use_gating
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        
        # 多模态差分编码
        self.abs_diff_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.rel_diff_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制
        if self.use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        if self.use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid()
            )
        
        # 门控机制
        if self.use_gating:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
            
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor, 
                percep_feat: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            pre_feat: 前时相特征 [B, C, H, W]
            post_feat: 后时相特征 [B, C, H, W]
            percep_feat: 感知特征 [B, C, H, W]
            
        Returns:
            enhanced_percep: 增强后的感知特征 [B, C, H, W]
            spatial_att: 空间注意力图 (用于可视化)
        """
        # 多模态差分编码
        abs_diff = torch.abs(pre_feat - post_feat)
        abs_diff_encoded = self.abs_diff_conv(abs_diff)
        
        # 相对差异 (余弦相似度)
        pre_norm = F.normalize(pre_feat, p=2, dim=1)
        post_norm = F.normalize(post_feat, p=2, dim=1)
        cosine_sim = (pre_norm * post_norm).sum(dim=1, keepdim=True)
        rel_diff = (1 - cosine_sim).expand_as(pre_feat)
        rel_diff_encoded = self.rel_diff_conv(rel_diff)
        
        # 上下文信息
        context = self.context_conv(percep_feat)
        
        # 融合差分信息
        diff_info = abs_diff_encoded + rel_diff_encoded + context
        
        # 注意力增强
        spatial_att = None
        if self.use_channel_attention:
            channel_att = self.channel_attention(diff_info)
            diff_info = diff_info * channel_att
        
        if self.use_spatial_attention:
            spatial_att = self.spatial_attention(diff_info)
            diff_info = diff_info * spatial_att
        
        # 门控调制
        if self.use_gating:
            spatial_gate = self.spatial_gate(diff_info)
            channel_gate = self.channel_gate(diff_info)
            gated_diff = diff_info * spatial_gate * channel_gate
        else:
            gated_diff = diff_info
        
        # 残差融合
        enhanced_percep = percep_feat + self.fusion_conv(gated_diff)
        
        return enhanced_percep, spatial_att


class CrossLevelAttentionPropagation(nn.Module):
    """
    跨层级注意力传播机制
    
    功能: 在多个尺度之间传播语义信息
    """
    
    def __init__(self, 
                 channel_list: List[int],
                 use_top_down: bool = True,
                 use_bottom_up: bool = True):
        super().__init__()
        
        self.num_levels = len(channel_list)
        self.use_top_down = use_top_down
        self.use_bottom_up = use_bottom_up
        
        # 自上而下的注意力强化
        if self.use_top_down:
            self.top_down_convs = nn.ModuleList()
            for i in range(self.num_levels - 1):
                deep_channels = channel_list[i + 1]
                shallow_channels = channel_list[i]
                
                self.top_down_convs.append(
                    nn.Sequential(
                        nn.Conv2d(deep_channels, shallow_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(shallow_channels),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(shallow_channels, shallow_channels, kernel_size=3, padding=1),
                        nn.Sigmoid()
                    )
                )
        
        # 自下而上的注意力增强
        if self.use_bottom_up:
            self.bottom_up_convs = nn.ModuleList()
            for i in range(self.num_levels - 1):
                shallow_channels = channel_list[i]
                deep_channels = channel_list[i + 1]
                
                self.bottom_up_convs.append(
                    nn.Sequential(
                        nn.Conv2d(shallow_channels, deep_channels, kernel_size=3, 
                                 stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(deep_channels),
                        nn.Conv2d(deep_channels, deep_channels, kernel_size=3, padding=1),
                        nn.Sigmoid()
                    )
                )
        
        # 特征融合卷积
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for channels in channel_list
        ])
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: 各层级特征 [f1, f2, f3, f4]，从浅到深
            
        Returns:
            enhanced_features: 增强后的特征
            attention_maps: 各层级的注意力图
        """
        attention_maps = []
        
        # 自上而下传播
        if self.use_top_down:
            top_down_features = [features[-1]]
            
            for i in range(self.num_levels - 2, -1, -1):
                deep_feat = top_down_features[0]
                shallow_feat = features[i]
                
                cond_attention = self.top_down_convs[i](deep_feat)
                attention_maps.insert(0, cond_attention)
                
                enhanced_shallow = shallow_feat * (1 + cond_attention)
                top_down_features.insert(0, enhanced_shallow)
        else:
            top_down_features = features
            attention_maps = [None] * (self.num_levels - 1)
        
        # 自下而上传播
        if self.use_bottom_up:
            enhanced_features = [top_down_features[0]]
            
            for i in range(self.num_levels - 1):
                shallow_feat = enhanced_features[-1]
                deep_feat = top_down_features[i + 1]
                
                detail_attention = self.bottom_up_convs[i](shallow_feat)
                enhanced_deep = deep_feat * (1 + detail_attention)
                fused_feat = self.fusion_convs[i + 1](enhanced_deep)
                enhanced_features.append(fused_feat)
        else:
            enhanced_features = []
            for i in range(self.num_levels):
                fused_feat = self.fusion_convs[i](top_down_features[i])
                enhanced_features.append(fused_feat)
        
        return enhanced_features, attention_maps


class UnifiedEncoder(nn.Module):
    """
    统一编码器: 差分注意力 + 对比学习
    
    整合了两大改进:
    1. 差分注意力模块: 增强感知特征
    2. 对比学习分支: 优化特征表示
    
    工作流程:
    Input → X3D → DiffAttn → CrossLevel → [Task + Contrastive] Branches
    """
    
    def __init__(self, 
                 args: Any, 
                 embed_dims: List[int] = [24, 24, 48, 96],
                 # 差分注意力配置
                 use_differential_attention: bool = True,
                 use_gating: bool = True,
                 use_channel_attention: bool = True,
                 use_spatial_attention: bool = True,
                 use_cross_level: bool = True,
                 use_top_down: bool = True,
                 use_bottom_up: bool = True):
        """
        Args:
            args: 配置参数
            embed_dims: 各层级的嵌入维度 [24, 24, 48, 96]
            use_differential_attention: 是否使用差分注意力
            use_gating: 是否使用门控机制
            use_channel_attention: 是否使用通道注意力
            use_spatial_attention: 是否使用空间注意力
            use_cross_level: 是否使用跨层级传播
            use_top_down: 是否使用自上而下传播
            use_bottom_up: 是否使用自下而上传播
        """
        super().__init__()
        self.args = args
        self.embed_dims = embed_dims
        
        # 保存配置
        self.use_differential_attention = use_differential_attention
        self.use_cross_level = use_cross_level
        
        print(f"\n{'='*60}")
        print(f"Unified Encoder Configuration:")
        print(f"  Differential Attention: {use_differential_attention}")
        if use_differential_attention:
            print(f"    ├─ Gating: {use_gating}")
            print(f"    ├─ Channel Attention: {use_channel_attention}")
            print(f"    └─ Spatial Attention: {use_spatial_attention}")
        print(f"  Cross-Level Propagation: {use_cross_level}")
        if use_cross_level:
            print(f"    ├─ Top-Down: {use_top_down}")
            print(f"    └─ Bottom-Up: {use_bottom_up}")
        print(f"{'='*60}\n")
        
        # ========== Stage 1: X3D Backbone ==========
        self.x3d = create_x3d(input_clip_length=3, depth_factor=5.0)
        
        # 加载预训练权重
        try:
            state_dict = torch.load(args.pretrained, map_location='cpu')['model_state']
            msg = self.x3d.load_state_dict(state_dict, strict=True)
            print(f'✅ Loaded pretrained X3D: {args.pretrained}')
        except Exception as e:
            print(f'⚠️ Failed to load pretrained weights: {e}')
        
        # 可学习的感知帧
        self.perception_frames = nn.Parameter(
            torch.randn(1, 3, args.num_perception_frame, args.in_height, args.in_width),
            requires_grad=True
        )
        
        # 原始特征增强层
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ) for dim in embed_dims
        ])
        
        # ========== Stage 2: 差分注意力增强 ==========
        if self.use_differential_attention:
            self.diff_attention_modules = nn.ModuleList([
                DifferentialAttentionModule(
                    in_channels=dim,
                    reduction=16,
                    use_gating=use_gating,
                    use_channel_attention=use_channel_attention,
                    use_spatial_attention=use_spatial_attention
                )
                for dim in embed_dims
            ])
            print(f"✅ Initialized {len(self.diff_attention_modules)} DifferentialAttentionModules")
        
        # ========== Stage 3: 跨层级注意力传播 ==========
        if self.use_cross_level:
            self.cross_level_propagation = CrossLevelAttentionPropagation(
                channel_list=embed_dims,
                use_top_down=use_top_down,
                use_bottom_up=use_bottom_up
            )
            print(f"✅ Initialized CrossLevelAttentionPropagation")
        
        # 存储注意力图
        self.attention_maps = []
        
    def enhance(self, x: torch.Tensor, fc: nn.Module) -> torch.Tensor:
        """
        原始的时序差分增强
        
        Args:
            x: [B, C, T, H, W] - T=3 (pre, perception, post)
            fc: 增强卷积层
            
        Returns:
            enhanced_x: [B, C, T, H, W]
        """
        middle_idx = x.shape[2] // 2
        
        pre_frame = x[:, :, 0]
        post_frame = x[:, :, self.args.num_perception_frame + 1]
        
        temporal_diff = torch.abs(pre_frame - post_frame)
        enhancement_features = fc(temporal_diff)
        
        middle_frame = x[:, :, middle_idx]
        enhanced_middle_frame = middle_frame + enhancement_features
        
        enhanced_x = x.clone()
        enhanced_x[:, :, middle_idx] = enhanced_middle_frame
        
        return enhanced_x
    
    def base_forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        编码器前向传播
        
        Args:
            x: 输入帧序列 [B, 3, 3, H, W] (pre, perception, post)
            
        Returns:
            final_features: 最终的多尺度特征 (用于task + contrastive)
            pre_post_features: 前后时相特征 (用于对比学习的锚点计算)
        """
        out = []
        pre_post_features = []
        
        # ========== Stage 1: X3D基础特征提取 ==========
        for i in range(4):
            x = self.x3d.blocks[i](x)
            x = self.enhance(x, self.fc[i])
            
            # 提取各帧特征
            layer_feature = []
            for idx in range(self.args.num_perception_frame):
                layer_feature.append(x[:, :, idx + 1])
            
            pre_feat = x[:, :, 0]
            post_feat = x[:, :, self.args.num_perception_frame + 1]
            pre_post_features.append((pre_feat, post_feat))
            
            out.append(layer_feature)
        
        # ========== Stage 2: 差分注意力增强 ==========
        if self.use_differential_attention:
            enhanced_out = []
            attention_maps_per_layer = []
            
            for i in range(4):
                enhanced_layer = []
                layer_attention_maps = []
                
                for idx in range(self.args.num_perception_frame):
                    percep_feat = out[i][idx]
                    pre_feat, post_feat = pre_post_features[i]
                    
                    # 差分注意力增强
                    enhanced_feat, spatial_att = self.diff_attention_modules[i](
                        pre_feat, post_feat, percep_feat
                    )
                    
                    enhanced_layer.append(enhanced_feat)
                    if spatial_att is not None:
                        layer_attention_maps.append(spatial_att)
                
                enhanced_out.append(enhanced_layer)
                
                if layer_attention_maps:
                    avg_attention = torch.stack(layer_attention_maps, dim=0).mean(dim=0)
                    attention_maps_per_layer.append(avg_attention)
                else:
                    attention_maps_per_layer.append(None)
        else:
            enhanced_out = out
            attention_maps_per_layer = [None] * 4
        
        # ========== Stage 3: 跨层级注意力传播 ==========
        if self.use_cross_level:
            single_percep_features = [layer[0] for layer in enhanced_out]
            
            final_features_list, propagation_attention_maps = self.cross_level_propagation(
                single_percep_features
            )
            
            # 重新组织为列表格式
            final_features = []
            for i in range(4):
                layer_features = []
                for idx in range(self.args.num_perception_frame):
                    if idx == 0:
                        layer_features.append(final_features_list[i])
                    else:
                        layer_features.append(enhanced_out[i][idx])
                final_features.append(layer_features)
            
            # 合并注意力图
            self.attention_maps = attention_maps_per_layer + propagation_attention_maps
        else:
            final_features = enhanced_out
            self.attention_maps = attention_maps_per_layer
        
        return final_features, pre_post_features
    
    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        主前向传播接口
        
        Args:
            pre_img: 前时相图像 [B, 3, H, W]
            post_img: 后时相图像 [B, 3, H, W]
            
        Returns:
            final_features: 增强后的多尺度感知特征 [[f0], [f1], [f2], [f3]]
                          每个fi是 [B, Ci, Hi, Wi]
            pre_post_features: 前后时相特征 [(pre0, post0), ..., (pre3, post3)]
                              用于对比学习的锚点计算
        """
        # 扩展感知帧
        expand_percep_frames = repeat(
            self.perception_frames, '1 c t h w -> b c t h w', b=pre_img.shape[0]
        )
        
        # 组合输入 [B, 3, 3, H, W]
        frames = torch.cat([
            pre_img.unsqueeze(2),      # [B, 3, 1, H, W]
            expand_percep_frames,       # [B, 3, T, H, W]
            post_img.unsqueeze(2)       # [B, 3, 1, H, W]
        ], dim=2)
        
        # 前向传播
        final_features, pre_post_features = self.base_forward(frames)
        
        return final_features, pre_post_features
    
    def get_attention_maps(self) -> List[torch.Tensor]:
        """获取注意力图 (用于可视化和一致性损失)"""
        return self.attention_maps
    
    def clear_attention_maps(self):
        """清空注意力图缓存"""
        self.attention_maps = []


# ========== 辅助函数 ==========

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_encoder_summary(encoder: UnifiedEncoder):
    """打印编码器摘要"""
    total, trainable = count_parameters(encoder)
    
    print(f"\n{'='*60}")
    print(f"Unified Encoder Summary")
    print(f"{'='*60}")
    print(f"Total Parameters: {total/1e6:.2f}M")
    print(f"Trainable Parameters: {trainable/1e6:.2f}M")
    print(f"\nModule Breakdown:")
    print(f"  X3D Backbone: ~{count_parameters(encoder.x3d)[0]/1e6:.2f}M")
    
    if encoder.use_differential_attention:
        diff_params = sum(count_parameters(m)[0] for m in encoder.diff_attention_modules)
        print(f"  Differential Attention: ~{diff_params/1e3:.1f}K")
    
    if encoder.use_cross_level:
        cross_params = count_parameters(encoder.cross_level_propagation)[0]
        print(f"  Cross-Level Propagation: ~{cross_params/1e3:.1f}K")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    """测试编码器"""
    from argparse import Namespace
    
    # 模拟配置
    args = Namespace(
        pretrained='path/to/X3D_L.pyth',
        num_perception_frame=1,
        in_height=256,
        in_width=256
    )
    
    # 创建编码器
    encoder = UnifiedEncoder(
        args,
        embed_dims=[24, 24, 48, 96],
        use_differential_attention=True,
        use_gating=True,
        use_channel_attention=True,
        use_spatial_attention=True,
        use_cross_level=True,
        use_top_down=True,
        use_bottom_up=True
    )
    
    # 打印摘要
    print_encoder_summary(encoder)
    
    # 测试前向传播
    pre_img = torch.randn(2, 3, 256, 256)
    post_img = torch.randn(2, 3, 256, 256)
    
    final_features, pre_post_features = encoder(pre_img, post_img)
    
    print(f"Forward Pass Test:")
    print(f"  Input: pre_img {pre_img.shape}, post_img {post_img.shape}")
    print(f"\n  Output Features (4 scales):")
    for i, layer_feats in enumerate(final_features):
        feat = layer_feats[0]  # 取第一个perception frame
        print(f"    Level {i}: {feat.shape}")
    
    print(f"\n  Pre/Post Features (for anchors):")
    for i, (pre, post) in enumerate(pre_post_features):
        print(f"    Level {i}: pre {pre.shape}, post {post.shape}")
    
    print(f"\n✅ Encoder test passed!")