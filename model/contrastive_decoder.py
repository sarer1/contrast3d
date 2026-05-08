# Copyright (c) Contextrast-Enhanced Change3D
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from model.contrastive_modules import ContrastiveLearningModule


class ContrastiveChangeDecoder(nn.Module):
    """
    对比学习增强的变化检测解码器
    
    在原始解码器基础上集成CCL和BANE
    """
    
    def __init__(self, 
                 args,
                 in_dim: List[int] = [24, 24, 48, 96],
                 # 对比学习参数
                 use_contrastive: bool = True,
                 temperature: float = 0.07,
                 alpha: float = 0.1,
                 use_bane: bool = True,
                 bane_ratio: float = 0.5):
        """
        Args:
            args: 配置参数
            in_dim: 输入特征维度
            use_contrastive: 是否使用对比学习
            temperature: 对比学习温度
            alpha: 对比损失权重
            use_bane: 是否使用BANE
            bane_ratio: BANE采样比例
        """
        super().__init__()
        
        self.use_contrastive = use_contrastive
        
        c1_channel, c2_channel, c3_channel, c4_channel = in_dim
        
        # ========== 原始解码器部分 ==========
        # 上采样块（c4 -> c3）
        self.up_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, c3_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c3_channel, c3_channel, kernel_size=4, 
                             stride=2, padding=1)
        )
        
        # 上采样块（c3 -> c2）
        self.up_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, c2_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, 
                             stride=2, padding=1)
        )
        
        # 上采样块（c2 -> c1）
        self.up_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, c1_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c1_channel, c1_channel, kernel_size=4, 
                             stride=2, padding=1)
        )
        
        # 最终预测层
        self.up_c1 = nn.Sequential(
            nn.Conv2d(c1_channel, 1, kernel_size=3, stride=1, 
                     padding=1, bias=False)
        )
        
        # ========== 对比学习模块 ==========
        if self.use_contrastive:
            self.contrastive_module = ContrastiveLearningModule(
                embed_dims=in_dim,
                num_classes=2,  # 二值变化检测
                temperature=temperature,
                alpha=alpha,
                use_bane=use_bane,
                bane_ratio=bane_ratio
            )
        
    def decode(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        解码器前向传播
        
        Args:
            features: [c1, c2, c3, c4]
            
        Returns:
            prediction: [B, 1, H, W]
        """
        c1, c2, c3, c4 = features
        
        # 渐进式上采样
        c3f = c3 + self.up_c4(c4)
        c2f = c2 + self.up_c3(c3f)
        c1f = c1 + self.up_c2(c2f)
        
        # 最终预测
        pred = self.up_c1(c1f)
        pred = torch.sigmoid(pred)
        
        return pred
    
    def forward(self, 
                features: List[torch.Tensor],
                labels: torch.Tensor = None,
                return_loss: bool = False) -> Tuple:
        """
        前向传播
        
        Args:
            features: 多尺度特征
            labels: 标签（训练时需要）
            return_loss: 是否返回对比损失
            
        Returns:
            prediction: 预测结果
            contrastive_loss: 对比损失（如果return_loss=True）
            info_dict: 信息字典
        """
        # 解码得到预测
        prediction = self.decode(features)
        
        # 如果不需要对比学习或者是推理阶段
        if not self.use_contrastive or not return_loss or labels is None:
            return prediction, None, {}
        
        # 计算对比学习损失
        contrastive_loss, info_dict = self.contrastive_module(
            features,
            prediction.detach(),  # 不对预测梯度回传
            labels
        )
        
        return prediction, contrastive_loss, info_dict