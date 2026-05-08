#!/usr/bin/env python
# Copyright (c) Unified Framework Training Script
# All rights reserved.

"""
统一框架训练脚本

整合差分注意力和BCD对比学习的完整训练流程

使用方法:
    # 基础训练
    python train_unified.py --config unified
    
    # 消融实验
    python train_unified.py --config U0  # Baseline
    python train_unified.py --config U4  # Full model
    
    # 自定义参数
    python train_unified.py --config U4 --batch_size 12 --contrast_weight 0.2
"""

import os
import sys
import time
import math
import numpy as np
from os.path import join as osp
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt

# 添加项目路径
sys.path.insert(0, '.')

# 导入配置
from config_unified import parse_args, print_config, validate_config

# 导入数据
import data.dataset as RSDataset
import data.transforms as RSTransforms

# 导入模型
from unified_trainer import UnifiedTrainer, print_model_summary

# 导入工具
from model.utils import BCEDiceLoss, setup_logger, AverageMeter
from utils.metric_tool import ConfuseMatrixMeter


def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 确保可复现性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_loaders(args, train_transform, val_transform):
    """创建数据加载器"""
    print("\n" + "="*60)
    print("Creating Data Loaders")
    print("="*60)
    
    # 训练集
    train_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="train",
        transform=train_transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # 验证集
    val_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="val",
        transform=val_transform
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # 测试集
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    max_batches = len(train_loader)
    
    print(f"Train: {len(train_data)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_data)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_data)} samples, {len(test_loader)} batches")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader, max_batches


def create_optimizer(args, model):
    """创建优化器"""
    if args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    print(f"✅ Created optimizer: {args.optimizer}")
    return optimizer


def create_scheduler(args, optimizer, max_batches):
    """创建学习率调度器"""
    if args.scheduler_type == 'none':
        return None
    
    # 计算总更新次数
    accum = args.gradient_accumulation_steps
    updates_per_epoch = math.ceil(max_batches / accum)
    total_updates = math.ceil(args.max_steps / accum)
    
    # Warmup设置
    warmup_iter_steps = getattr(args, "warmup_steps", 1000)
    warmup_updates = math.ceil(warmup_iter_steps / accum)
    
    # 获取基础学习率
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    base_lr_max = max(base_lrs)
    
    if args.scheduler_type in ['poly', 'warmup_poly']:
        power = args.poly_power
        min_lr = args.min_lr
        
        def lr_lambda(update_step: int):
            # Warmup阶段
            if warmup_updates > 0 and update_step < warmup_updates:
                warm = (update_step + 1) / warmup_updates
                return 0.1 + 0.9 * warm
            
            # Poly衰减阶段
            t = min(update_step, total_updates)
            poly = (1 - t / total_updates) ** power
            
            # 最小学习率
            if min_lr > 0:
                min_ratio = min_lr / base_lr_max
                poly = max(poly, min_ratio)
            
            return poly
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        print(f"✅ Created scheduler: {args.scheduler_type}")
        print(f"   Total updates: {total_updates}")
        print(f"   Warmup updates: {warmup_updates}")
        print(f"   Poly power: {power}")
        print(f"   Min LR: {min_lr}")
        
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler_type}")
    
    return scheduler


@torch.no_grad()
def validate(args, val_loader, model, epoch, use_amp=True):
    """验证函数"""
    model.eval()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)
    
    print(f"\n{'='*60}")
    print(f"Validation (Epoch {epoch})")
    print(f"{'='*60}")
    
    for iter_idx, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        
        pre_img = img[:, 0:3].cuda(non_blocking=True).float()
        post_img = img[:, 3:6].cuda(non_blocking=True).float()
        target = target.cuda(non_blocking=True).float()

        start_time = time.time()

        # 前向传播
        if use_amp:
            with autocast():
                pred, _, _ = model(pre_img, post_img)
                loss = BCEDiceLoss(pred, target)
        else:
            pred, _, _ = model(pre_img, post_img)
            loss = BCEDiceLoss(pred, target)

        # 二值化预测
        pred_binary = torch.where(
            pred > 0.5,
            torch.ones_like(pred),
            torch.zeros_like(pred)
        ).long()

        time_taken = time.time() - start_time
        epoch_loss.append(loss.data.item())

        # 更新评估指标
        f1 = eval_meter.update_cm(
            pr=pred_binary.cpu().numpy(),
            gt=target.cpu().numpy()
        )
        
        if iter_idx % 50 == 0:
            print(
                f"\r[{iter_idx}/{total_batches}] "
                f"F1: {f1:.4f} | Loss: {loss.data.item():.4f} | "
                f"Time: {time_taken:.3f}s",
                end=''
            )

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()
    
    print()
    print(f"{'='*60}")
    print(f"Validation Results (Epoch {epoch}):")
    print(f"  Loss: {average_epoch_loss_val:.4f}")
    print(f"  F1: {scores['F1']:.4f} | IoU: {scores['IoU']:.4f}")
    print(f"  Kappa: {scores['Kappa']:.4f}")
    print(f"  Recall: {scores['recall']:.4f} | Precision: {scores['precision']:.4f}")
    print(f"{'='*60}\n")

    return average_epoch_loss_val, scores


def train_epoch(args, train_loader, model, optimizer, scaler, scheduler, 
                epoch, max_batches, cur_iter=0, use_amp=True):
    """训练一个epoch"""
    model.train()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    
    # 损失记录
    task_losses = AverageMeter()
    contrast_losses = AverageMeter()
    total_losses = AverageMeter()
    
    accumulation_steps = args.gradient_accumulation_steps

    for iter_idx, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs

        pre_img = img[:, 0:3].cuda(non_blocking=True).float()
        post_img = img[:, 3:6].cuda(non_blocking=True).float()
        target = target.cuda(non_blocking=True).float()

        start_time = time.time()

        # ========== 前向传播 ==========
        if use_amp:
            with autocast():
                # 模型前向
                pred, contrast_loss, _ = model(pre_img, post_img, target)
                
                # 任务损失
                task_loss = BCEDiceLoss(pred, target)
                
                # 总损失
                total_loss, contrast_weighted = model.get_total_loss(
                    task_loss, contrast_loss
                )
                
                # 梯度累积
                total_loss = total_loss / accumulation_steps
        else:
            pred, contrast_loss, _ = model(pre_img, post_img, target)
            task_loss = BCEDiceLoss(pred, target)
            total_loss, contrast_weighted = model.get_total_loss(
                task_loss, contrast_loss
            )
            total_loss = total_loss / accumulation_steps

        # 二值化预测
        pred_binary = torch.where(
            pred > 0.5,
            torch.ones_like(pred),
            torch.zeros_like(pred)
        ).long()

        # ========== 反向传播 ==========
        if use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # ========== 优化器步骤 ==========
        if (iter_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()

        # ========== 记录损失 ==========
        batch_size = pre_img.size(0)
        task_losses.update(task_loss.data.item(), batch_size)
        if contrast_loss is not None:
            contrast_losses.update(contrast_loss.data.item(), batch_size)
        total_losses.update(total_loss.data.item() * accumulation_steps, batch_size)
        
        # ========== 更新指标 ==========
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter_idx - cur_iter) * time_taken / 3600

        with torch.no_grad():
            f1 = eval_meter.update_cm(
                pr=pred_binary.cpu().numpy(),
                gt=target.cpu().numpy()
            )

        # ========== 打印日志 ==========
        if (iter_idx + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            global_step = cur_iter + iter_idx
            
            print(
                f"[Epoch {epoch}] [Iter {iter_idx + 1}/{len(train_loader)}] "
                f"[Step {global_step}] [ETA {res_time:.2f}h]\n"
                f"  LR: {current_lr:.6f} | "
                f"Task Loss: {task_losses.avg:.4f} | "
                f"Contrast Loss: {contrast_losses.avg:.4f} | "
                f"Total Loss: {total_losses.avg:.4f}\n"
                f"  F1: {f1:.4f} | "
                f"GPU Mem: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
            )

    # ========== Epoch结束处理剩余梯度 ==========
    if len(train_loader) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        optimizer.zero_grad()

    # ========== 返回epoch统计 ==========
    scores = eval_meter.get_scores()
    
    return {
        'task_loss': task_losses.avg,
        'contrast_loss': contrast_losses.avg,
        'total_loss': total_losses.avg,
        'scores': scores
    }


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    """加载checkpoint"""
    print(f"\n{'='*60}")
    print(f"📄 Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # 加载模型
    model.load_state_dict(checkpoint['state_dict'])
    print("✅ Model state loaded")
    
    # 加载优化器
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("✅ Optimizer state loaded")
    
    # 加载调度器
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("✅ Scheduler state loaded")
    
    # 加载scaler
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        print("✅ Scaler state loaded")
    
    # 提取恢复信息
    resume_info = {
        'start_epoch': checkpoint.get('epoch', 0),
        'best_F1': checkpoint.get('best_F1', 0.0),
        'global_update_step': checkpoint.get('global_update_step', 0),
    }
    
    print(f"\nResume Information:")
    print(f"  Start Epoch: {resume_info['start_epoch']}")
    print(f"  Best F1: {resume_info['best_F1']:.4f}")
    print(f"  Global Steps: {resume_info['global_update_step']}")
    print(f"{'='*60}\n")
    
    return resume_info


def save_checkpoint(save_path, epoch, model, optimizer, scheduler, scaler, 
                   metrics, is_best=False):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict(),
        **metrics
    }
    
    # 保存最新checkpoint
    checkpoint_path = osp(save_path, 'checkpoint.pth.tar')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = osp(save_path, 'best_model.pth')
        torch.save(model.state_dict(), best_path)
        print(f"💾 Saved best model to: {best_path}")


def main():
    """主训练函数"""
    # ========== 解析配置 ==========
    args = parse_args()
    print_config(args)
    validate_config(args)
    
    # ========== 设置环境 ==========
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    # ========== 创建保存目录 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ========== 初始化模型 ==========
    print(f"\n{'='*60}")
    print("Initializing Model")
    print(f"{'='*60}")
    
    model = UnifiedTrainer(args).cuda().float()
    print_model_summary(model)
    
    # ========== 数据加载 ==========
    train_transform, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(
        args, train_transform, val_transform
    )
    
    # 计算最大epochs
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    print(f"Max epochs: {args.max_epochs} (based on {args.max_steps} steps)")
    
    # ========== 优化器和调度器 ==========
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer, max_batches)
    scaler = GradScaler(enabled=args.use_amp)
    
    # ========== 设置日志 ==========
    logger = setup_logger(args, args.save_dir)
    
    # ========== 恢复训练 ==========
    start_epoch = 0
    best_F1 = 0.0
    early_stop_counter = 0
    global_update_step = 0
    
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        default_checkpoint = osp(args.save_dir, 'checkpoint.pth.tar')
        if os.path.exists(default_checkpoint):
            resume_path = default_checkpoint
    
    if resume_path and os.path.exists(resume_path):
        resume_info = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        start_epoch = resume_info['start_epoch']
        best_F1 = resume_info['best_F1']
        global_update_step = resume_info['global_update_step']
    
    # ========== 主训练循环 ==========
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()
        epoch_start = time.time()
        
        # ========== 训练 ==========
        train_stats = train_epoch(
            args, train_loader, model, optimizer, scaler, scheduler,
            epoch, max_batches, cur_iter=epoch * max_batches,
            use_amp=args.use_amp
        )
        
        # 更新全局步数
        accum = args.gradient_accumulation_steps
        global_update_step += math.ceil(max_batches / accum)
        
        # ========== 验证 ==========
        if epoch == 0 and start_epoch == 0:
            print(f"\n⏭️  Skipping validation for epoch 0")
            continue
        
        if (epoch + 1) % args.val_freq == 0:
            torch.cuda.empty_cache()
            val_loss, val_scores = validate(
                args, test_loader, model, epoch, use_amp=args.use_amp
            )
            
            current_F1 = val_scores['F1']
            epoch_time = time.time() - epoch_start
            
            # ========== 记录日志 ==========
            logger.write(
                f"\n{epoch}\t\t{val_scores['Kappa']:.4f}\t\t"
                f"{val_scores['IoU']:.4f}\t\t{current_F1:.4f}\t\t"
                f"{val_scores['recall']:.4f}\t\t{val_scores['precision']:.4f}"
            )
            logger.flush()
            
            # ========== 保存checkpoint ==========
            metrics = {
                'loss_train': train_stats['total_loss'],
                'loss_val': val_loss,
                'task_loss': train_stats['task_loss'],
                'contrast_loss': train_stats['contrast_loss'],
                'F_train': train_stats['scores']['F1'],
                'F_val': current_F1,
                'best_F1': best_F1,
                'global_update_step': global_update_step,
            }
            
            is_best = current_F1 > best_F1 + args.min_delta
            
            save_checkpoint(
                args.save_dir, epoch, model, optimizer, scheduler, scaler,
                metrics, is_best
            )
            
            # ========== 早停逻辑 ==========
            if is_best:
                best_F1 = current_F1
                early_stop_counter = 0
                best_epoch = epoch
                
                print(f"\n🎯 New Best Model!")
                print(f"  Epoch: {epoch}")
                print(f"  Validation F1: {best_F1:.4f}")
            else:
                early_stop_counter += 1
                print(f"\n⏸️  No improvement")
                print(f"  Current F1: {current_F1:.4f}")
                print(f"  Best F1: {best_F1:.4f}")
                print(f"  Patience: {early_stop_counter}/{args.patience}")
            
            # 检查早停
            if early_stop_counter >= args.patience:
                total_time = time.time() - training_start_time
                print(f"\n{'='*60}")
                print(f"🛑 Early Stopping Triggered!")
                print(f"  Best Epoch: {best_epoch}")
                print(f"  Best F1: {best_F1:.4f}")
                print(f"  Total Time: {total_time / 3600:.2f}h")
                print(f"{'='*60}\n")
                break
            
            # ========== Epoch总结 ==========
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Time: {epoch_time / 60:.2f} min")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_stats['total_loss']:.4f} "
                  f"(Task: {train_stats['task_loss']:.4f}, "
                  f"Contrast: {train_stats['contrast_loss']:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Train F1: {train_stats['scores']['F1']:.4f} | "
                  f"Val F1: {current_F1:.4f}")
            print(f"  Best F1: {best_F1:.4f} (Epoch {best_epoch if is_best else 'previous'})")
            print(f"{'='*60}\n")
    
    # ========== 最终测试 ==========
    print(f"\n{'='*60}")
    print("Final Testing with Best Model")
    print(f"{'='*60}")
    
    best_model_path = osp(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("⚠️  Best model not found, using last checkpoint")
    
    test_loss, test_scores = validate(args, test_loader, model, 0, use_amp=args.use_amp)
    
    print(f"\n{'='*60}")
    print("📊 Final Test Results:")
    print(f"{'='*60}")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Kappa: {test_scores['Kappa']:.4f}")
    print(f"  IoU: {test_scores['IoU']:.4f}")
    print(f"  F1: {test_scores['F1']:.4f}")
    print(f"  Recall: {test_scores['recall']:.4f}")
    print(f"  Precision: {test_scores['precision']:.4f}")
    print(f"{'='*60}\n")
    
    logger.write(
        f"\nTest\t\t{test_scores['Kappa']:.4f}\t\t"
        f"{test_scores['IoU']:.4f}\t\t{test_scores['F1']:.4f}\t\t"
        f"{test_scores['recall']:.4f}\t\t{test_scores['precision']:.4f}"
    )
    logger.flush()
    logger.close()
    
    total_training_time = time.time() - training_start_time
    print(f"✅ Training completed in {total_training_time / 3600:.2f} hours")
    print(f"📁 Results saved to: {args.save_dir}\n")


if __name__ == '__main__':
    main()