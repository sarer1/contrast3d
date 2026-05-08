# Copyright (c) Poly Learning Rate Scheduler
# All rights reserved.

import math
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRScheduler(_LRScheduler):
    """
    多项式学习率调度器
    
    lr = base_lr * (1 - iter/max_iter)^power
    """
    
    def __init__(self, 
                 optimizer,
                 max_iterations: int,
                 power: float = 0.9,
                 warmup_iterations: int = 0,
                 warmup_start_lr: float = 1e-6,
                 min_lr: float = 0,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: 优化器
            max_iterations: 最大迭代次数
            power: 多项式幂次
            warmup_iterations: 预热迭代次数
            warmup_start_lr: 预热起始学习率
            min_lr: 最小学习率
            last_epoch: 上次epoch（用于续训）
        """
        self.max_iterations = max_iterations
        self.power = power
        self.warmup_iterations = warmup_iterations
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前学习率"""
        if self.last_epoch < self.warmup_iterations:
            # 预热阶段：线性增长
            alpha = self.last_epoch / self.warmup_iterations
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # 多项式衰减阶段
            progress = (self.last_epoch - self.warmup_iterations) / \
                      (self.max_iterations - self.warmup_iterations)
            progress = min(progress, 1.0)
            
            return [
                max(self.min_lr, base_lr * (1 - progress) ** self.power)
                for base_lr in self.base_lrs
            ]