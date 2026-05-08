#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试与可视化脚本
功能：
1. 加载训练好的权重评估测试集
2. 计算完整评价指标（Kappa、OA、Precision、F1、Recall、IoU）
3. 生成指定颜色规则的预测可视化图并保存
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from os.path import join as osp

# 添加项目路径
sys.path.insert(0, '.')

# 导入必要模块
from config_unified import parse_args, print_config
from data.dataset import BCDDataset
from data.transforms import BCDTransforms
from unified_trainer import UnifiedTrainer
from model.utils import setup_logger
from utils.metric_tool import ConfuseMatrixMeter

# 颜色映射配置 (BGR格式，适配OpenCV)
COLOR_MAP = {
    'unpredicted_change': (0, 0, 255),    # 未预测的变化 - 红色
    'false_predicted_change': (0, 255, 0), # 错误预测的变化 - 绿色
    'correct_change': (255, 255, 255),     # 正确预测的变化 - 白色
    'correct_no_change': (0, 0, 0)         # 正确预测的未变化 - 黑色
}

def set_env(args):
    """设置环境"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def load_model(args):
    """加载模型和权重"""
    print("="*60)
    print("Loading model and weights")
    print("="*60)
    
    # 初始化模型
    model = UnifiedTrainer(args).cuda().float()
    
    # 加载权重
    weight_path = args.weight_path
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    
    if weight_path.endswith('.pth'):
        # 加载纯模型权重
        state_dict = torch.load(weight_path, map_location='cuda')
        model.load_state_dict(state_dict)
    elif weight_path.endswith('.pth.tar'):
        # 加载checkpoint
        checkpoint = torch.load(weight_path, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f"Unsupported weight format: {weight_path}")
    
    model.eval()
    print(f"✅ Successfully loaded weights from {weight_path}")
    return model

def create_test_loader(args):
    """创建测试集加载器"""
    print("\n" + "="*60)
    print("Creating Test Data Loader")
    print("="*60)
    
    # 获取测试变换（无数据增强）
    _, val_transform = BCDTransforms.get_transform_pipelines(args)
    
    # 测试集
    test_data = BCDDataset(
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
    
    print(f"Test: {len(test_data)} samples, {len(test_loader)} batches")
    print("="*60 + "\n")
    
    return test_loader, test_data

def calculate_metrics(pred_binary, target):
    """
    计算完整的评价指标
    Args:
        pred_binary: 预测二值图 (numpy array, shape [B, H, W], 0/1)
        target: 真实标签 (numpy array, shape [B, H, W], 0/1)
    Returns:
        dict: 包含所有指标的字典
    """
    # 将预测和标签展平
    pred_flat = pred_binary.reshape(-1)
    target_flat = target.reshape(-1)
    
    # 计算混淆矩阵元素
    TP = np.sum((pred_flat == 1) & (target_flat == 1))  # 真阳性
    TN = np.sum((pred_flat == 0) & (target_flat == 0))  # 真阴性
    FP = np.sum((pred_flat == 1) & (target_flat == 0))  # 假阳性
    FN = np.sum((pred_flat == 0) & (target_flat == 1))  # 假阴性
    
    # 计算各项指标
    metrics = {}
    
    # 总体精度 OA (Overall Accuracy)
    total = TP + TN + FP + FN
    metrics['OA'] = (TP + TN) / total if total > 0 else 0.0
    
    # Kappa系数
    po = metrics['OA']
    pe = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / (total ** 2)
    metrics['Kappa'] = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    
    # Precision (精确率)
    metrics['Precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Recall (召回率)
    metrics['Recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # F1 分数
    precision = metrics['Precision']
    recall = metrics['Recall']
    metrics['F1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # IoU (交并比)
    metrics['IoU'] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    
    # 保存混淆矩阵元素
    metrics['confusion_matrix'] = {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }
    
    return metrics

def create_visualization(pred_binary, target, img_shape):
    """
    创建指定颜色规则的可视化图像
    Args:
        pred_binary: 预测二值图 (numpy array, shape [H, W], 0/1)
        target: 真实标签 (numpy array, shape [H, W], 0/1)
        img_shape: 输出图像形状 (H, W)
    Returns:
        numpy array: 可视化图像 (H, W, 3)
    """
    # 初始化可视化图像
    vis_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    
    # 定义各个区域的掩码
    # 1. 正确预测的变化 (TP): 真实=1，预测=1 → 白色
    mask_correct_change = (target == 1) & (pred_binary == 1)
    vis_img[mask_correct_change] = COLOR_MAP['correct_change']
    
    # 2. 正确预测的未变化 (TN): 真实=0，预测=0 → 黑色
    mask_correct_no_change = (target == 0) & (pred_binary == 0)
    vis_img[mask_correct_no_change] = COLOR_MAP['correct_no_change']
    
    # 3. 错误预测的变化 (FP): 真实=0，预测=1 → 绿色
    mask_false_change = (target == 0) & (pred_binary == 1)
    vis_img[mask_false_change] = COLOR_MAP['false_predicted_change']
    
    # 4. 未预测的变化 (FN): 真实=1，预测=0 → 红色
    mask_unpredicted_change = (target == 1) & (pred_binary == 0)
    vis_img[mask_unpredicted_change] = COLOR_MAP['unpredicted_change']
    
    return vis_img

@torch.no_grad()
def test_and_visualize(args, model, test_loader, test_data):
    """
    执行测试并生成可视化结果
    """
    print("="*60)
    print("Starting Test and Visualization")
    print("="*60)
    
    # 创建可视化保存目录
    vis_dir = osp(args.save_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 初始化指标统计
    all_preds = []
    all_targets = []
    
    # 逐个处理测试样本
    for batch_idx, batched_inputs in enumerate(test_loader):
        img, target = batched_inputs
        
        # 数据移到GPU
        pre_img = img[:, 0:3].cuda(non_blocking=True).float()
        post_img = img[:, 3:6].cuda(non_blocking=True).float()
        target_np = target.numpy()
        
        # 模型前向传播
        pred, _, _ = model(pre_img, post_img)
        
        # 二值化预测
        pred_binary = torch.where(
            pred > 0.5,
            torch.ones_like(pred),
            torch.zeros_like(pred)
        ).cpu().numpy().astype(np.uint8)
        
        # 收集所有预测和标签
        all_preds.append(pred_binary)
        all_targets.append(target_np)
        
        # 生成并保存可视化结果
        for sample_idx in range(pred_binary.shape[0]):
            # 计算全局样本索引
            global_idx = batch_idx * args.batch_size + sample_idx
            if global_idx >= len(test_data):
                break
            
            # 获取单样本数据
            pred_sample = pred_binary[sample_idx, 0]  # [H, W]
            target_sample = target_np[sample_idx, 0]  # [H, W]
            
            # 创建可视化图像
            vis_img = create_visualization(pred_sample, target_sample, pred_sample.shape)
            
            # 保存可视化图像
            vis_path = osp(vis_dir, f"test_{global_idx:04d}_vis.png")
            cv2.imwrite(vis_path, vis_img)
            
            if global_idx % 1 == 0:
                print(f"✅ Saved visualization for sample {global_idx} to {vis_path}")
    
    # 合并所有预测和标签
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算整体指标
    overall_metrics = calculate_metrics(all_preds[:, 0], all_targets[:, 0])
    
    # 打印完整指标
    print("\n" + "="*60)
    print("📊 Final Test Metrics")
    print("="*60)
    print(f"Overall Accuracy (OA):    {overall_metrics['OA']:.4f}")
    print(f"Kappa Coefficient:        {overall_metrics['Kappa']:.4f}")
    print(f"Precision:                {overall_metrics['Precision']:.4f}")
    print(f"Recall:                   {overall_metrics['Recall']:.4f}")
    print(f"F1 Score:                 {overall_metrics['F1']:.4f}")
    print(f"IoU:                      {overall_metrics['IoU']:.4f}")
    print("="*60)
    
    # 保存指标到文件
    metrics_path = osp(args.save_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Test Set Evaluation Metrics\n")
        f.write("="*50 + "\n")
        f.write(f"OA:        {overall_metrics['OA']:.4f}\n")
        f.write(f"Kappa:     {overall_metrics['Kappa']:.4f}\n")
        f.write(f"Precision: {overall_metrics['Precision']:.4f}\n")
        f.write(f"Recall:    {overall_metrics['Recall']:.4f}\n")
        f.write(f"F1:        {overall_metrics['F1']:.4f}\n")
        f.write(f"IoU:       {overall_metrics['IoU']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"TP: {overall_metrics['confusion_matrix']['TP']}\n")
        f.write(f"TN: {overall_metrics['confusion_matrix']['TN']}\n")
        f.write(f"FP: {overall_metrics['confusion_matrix']['FP']}\n")
        f.write(f"FN: {overall_metrics['confusion_matrix']['FN']}\n")
    
    print(f"\n✅ Metrics saved to {metrics_path}")
    print(f"✅ Visualizations saved to {vis_dir}")
    
    return overall_metrics

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 添加测试专用参数
    args.weight_path = getattr(args, 'weight_path', 'exp_unified/U3_contrastive_only_sysu/best_model.pth')
    args.save_dir = getattr(args, 'save_dir', 'exp_unified/U3_contrastive_only_sysu/test_results')
    args.batch_size = getattr(args, 'batch_size', 1)  # 可视化建议batch_size=1
    args.num_workers = getattr(args, 'num_workers', 8)
    
    # 检查必要参数
    if args.weight_path is None:
        raise ValueError("Please specify weight path with --weight_path")
    
    # 打印配置
    print_config(args)
    
    # 设置环境
    set_env(args)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args)
    
    # 创建测试加载器
    test_loader, test_data = create_test_loader(args)
    
    # 执行测试和可视化
    test_and_visualize(args, model, test_loader, test_data)
    
    print("\n🎉 Test and visualization completed successfully!")

if __name__ == '__main__':
    main()