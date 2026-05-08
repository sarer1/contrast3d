# Copyright (c) Checkpoint Manager
# All rights reserved.

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """
    检查点管理器
    
    支持保存、加载、续训
    """
    
    def __init__(self, 
                 save_dir: str,
                 max_checkpoints: int = 5):
        """
        Args:
            save_dir: 保存目录
            max_checkpoints: 最多保留的检查点数量
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(self,
                       epoch: int,
                       iteration: int,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Any,
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       extra_state: Optional[Dict] = None):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            iteration: 当前迭代次数
            model: 模型
            optimizer: 优化器
            scheduler: 调度器
            metrics: 指标字典
            is_best: 是否为最佳模型
            extra_state: 额外状态
        """
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
        }
        
        if extra_state is not None:
            checkpoint.update(extra_state)
        
        # 保存最新检查点
        latest_path = self.save_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        print(f'✓ Saved latest checkpoint to {latest_path}')
        
        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f'✓ Saved best checkpoint to {best_path}')
        
        # 定期保存
        if epoch % 10 == 0:
            epoch_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
            print(f'✓ Saved epoch checkpoint to {epoch_path}')
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def load_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       checkpoint_path: Optional[str] = None,
                       load_best: bool = False) -> Dict:
        """
        加载检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 调度器
            checkpoint_path: 检查点路径
            load_best: 是否加载最佳模型
            
        Returns:
            checkpoint: 检查点字典
        """
        # 确定加载路径
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        elif load_best:
            path = self.save_dir / 'checkpoint_best.pth'
        else:
            path = self.save_dir / 'checkpoint_latest.pth'
        
        if not path.exists():
            print(f'✗ Checkpoint not found: {path}')
            return {}
        
        # 加载检查点
        print(f'Loading checkpoint from {path}...')
        checkpoint = torch.load(path, map_location='cpu')
        
        # 加载模型
        model.load_state_dict(checkpoint['model_state_dict'])
        print('✓ Loaded model state')
        
        # 加载优化器
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('✓ Loaded optimizer state')
        
        # 加载调度器
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('✓ Loaded scheduler state')
        
        print(f"✓ Resumed from epoch {checkpoint.get('epoch', 0)}, "
              f"iteration {checkpoint.get('iteration', 0)}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """清理旧的epoch检查点"""
        epoch_checkpoints = sorted(
            self.save_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        if len(epoch_checkpoints) > self.max_checkpoints:
            for old_ckpt in epoch_checkpoints[:-self.max_checkpoints]:
                old_ckpt.unlink()
                print(f'✓ Removed old checkpoint: {old_ckpt.name}')