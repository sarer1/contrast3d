# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# 模型测试、评价指标、可视化、FLOPs 和参数量统计脚本

# 功能：
# 1. 加载训练好的权重并评估测试集
# 2. 计算完整评价指标：Kappa、OA、Precision、F1、Recall、IoU
# 3. 生成指定颜色规则的预测可视化图并保存
# 4. 计算模型 FLOPs 和参数量
# """

# import os
# import sys
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import cv2
# from os.path import join as osp

# # FLOPs / Params 统计工具
# from thop import profile, clever_format

# # 添加项目路径
# sys.path.insert(0, ".")

# # 导入项目模块
# from config_unified import parse_args, print_config
# from data.dataset import BCDDataset
# from data.transforms import BCDTransforms
# from unified_trainer import UnifiedTrainer


# # 颜色映射配置，BGR 格式，适配 OpenCV
# COLOR_MAP = {
#     "unpredicted_change": (0, 0, 255),       # FN：未预测出的变化，红色
#     "false_predicted_change": (0, 255, 0),   # FP：错误预测为变化，绿色
#     "correct_change": (255, 255, 255),       # TP：正确预测的变化，白色
#     "correct_no_change": (0, 0, 0),          # TN：正确预测的未变化，黑色
# }


# def set_env(args):
#     """设置运行环境"""

#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True


# def get_device():
#     """获取设备"""

#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     return torch.device("cpu")


# def load_model(args, device):
#     """加载模型和权重"""

#     print("=" * 60)
#     print("Loading model and weights")
#     print("=" * 60)

#     model = UnifiedTrainer(args).to(device).float()

#     weight_path = args.weight_path

#     if not os.path.exists(weight_path):
#         raise FileNotFoundError(f"Weight file not found: {weight_path}")

#     checkpoint = torch.load(weight_path, map_location=device)

#     if weight_path.endswith(".pth"):
#         # 情况 1：直接保存的是 state_dict
#         if isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
#             model.load_state_dict(checkpoint)

#         # 情况 2：虽然是 .pth，但里面是 checkpoint 格式
#         elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#             model.load_state_dict(checkpoint["state_dict"])

#         else:
#             raise ValueError("Unsupported .pth weight format.")

#     elif weight_path.endswith(".pth.tar"):
#         if "state_dict" not in checkpoint:
#             raise KeyError("Checkpoint does not contain key: state_dict")
#         model.load_state_dict(checkpoint["state_dict"])

#     else:
#         raise ValueError(f"Unsupported weight format: {weight_path}")

#     model.eval()

#     print(f"✅ Successfully loaded weights from {weight_path}")

#     return model


# class ModelWrapper(nn.Module):
#     """
#     FLOPs 统计包装器

#     原始模型 forward 返回：
#         pred, _, _ = model(pre_img, post_img)

#     thop 统计时只需要主预测结果 pred。
#     """

#     def __init__(self, model):
#         super(ModelWrapper, self).__init__()
#         self.model = model

#     def forward(self, pre_img, post_img):
#         output = self.model(pre_img, post_img)

#         if isinstance(output, (tuple, list)):
#             return output[0]

#         return output


# def get_input_size(args):
#     """
#     获取 FLOPs 统计输入尺寸

#     优先级：
#     1. args.img_size
#     2. args.crop_size
#     3. args.input_size
#     4. 默认 256
#     """

#     if hasattr(args, "img_size"):
#         input_size = args.img_size
#     elif hasattr(args, "crop_size"):
#         input_size = args.crop_size
#     elif hasattr(args, "input_size"):
#         input_size = args.input_size
#     else:
#         input_size = 256

#     if isinstance(input_size, str):
#         # 支持 "256" 或 "256,256" 或 "256x256"
#         input_size = input_size.lower().replace("x", ",")
#         parts = [int(x.strip()) for x in input_size.split(",") if x.strip()]

#         if len(parts) == 1:
#             h, w = parts[0], parts[0]
#         elif len(parts) >= 2:
#             h, w = parts[0], parts[1]
#         else:
#             h, w = 256, 256

#     elif isinstance(input_size, (tuple, list)):
#         if len(input_size) == 1:
#             h, w = input_size[0], input_size[0]
#         else:
#             h, w = input_size[0], input_size[1]

#     else:
#         h, w = int(input_size), int(input_size)

#     return int(h), int(w)


# def calculate_flops_and_params(args, model, device):
#     """计算模型 FLOPs 和参数量"""

#     print("\n" + "=" * 60)
#     print("Calculating FLOPs and Parameters")
#     print("=" * 60)

#     model.eval()

#     h, w = get_input_size(args)

#     pre_img = torch.randn(1, 3, h, w).to(device).float()
#     post_img = torch.randn(1, 3, h, w).to(device).float()

#     wrapped_model = ModelWrapper(model).to(device).eval()

#     with torch.no_grad():
#         flops, params = profile(
#             wrapped_model,
#             inputs=(pre_img, post_img),
#             verbose=False
#         )

#     flops_readable, params_readable = clever_format([flops, params], "%.3f")

#     print(f"Input size:              pre_img  = 1 x 3 x {h} x {w}")
#     print(f"Input size:              post_img = 1 x 3 x {h} x {w}")
#     print(f"FLOPs:                   {flops_readable}")
#     print(f"Parameters:              {params_readable}")
#     print("=" * 60)

#     flops_path = osp(args.save_dir, "model_flops_params.txt")

#     with open(flops_path, "w", encoding="utf-8") as f:
#         f.write("Model FLOPs and Parameters\n")
#         f.write("=" * 50 + "\n")
#         f.write(f"Input pre_img:  1 x 3 x {h} x {w}\n")
#         f.write(f"Input post_img: 1 x 3 x {h} x {w}\n")
#         f.write(f"FLOPs:          {flops_readable}\n")
#         f.write(f"Params:         {params_readable}\n")
#         f.write(f"Raw FLOPs:      {flops}\n")
#         f.write(f"Raw Params:     {params}\n")

#     print(f"✅ FLOPs and parameters saved to {flops_path}")

#     return flops, params


# def create_test_loader(args):
#     """创建测试集加载器"""

#     print("\n" + "=" * 60)
#     print("Creating Test Data Loader")
#     print("=" * 60)

#     _, val_transform = BCDTransforms.get_transform_pipelines(args)

#     test_data = BCDDataset(
#         file_root=args.file_root,
#         split="test",
#         transform=val_transform
#     )

#     test_loader = DataLoader(
#         test_data,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         persistent_workers=True if args.num_workers > 0 else False
#     )

#     print(f"Test: {len(test_data)} samples, {len(test_loader)} batches")
#     print("=" * 60 + "\n")

#     return test_loader, test_data


# def calculate_metrics(pred_binary, target):
#     """
#     计算完整评价指标

#     Args:
#         pred_binary: numpy array, shape [B, H, W], 值为 0/1
#         target: numpy array, shape [B, H, W], 值为 0/1

#     Returns:
#         dict
#     """

#     pred_flat = pred_binary.reshape(-1).astype(np.uint8)
#     target_flat = target.reshape(-1).astype(np.uint8)

#     TP = np.sum((pred_flat == 1) & (target_flat == 1))
#     TN = np.sum((pred_flat == 0) & (target_flat == 0))
#     FP = np.sum((pred_flat == 1) & (target_flat == 0))
#     FN = np.sum((pred_flat == 0) & (target_flat == 1))

#     total = TP + TN + FP + FN

#     metrics = {}

#     metrics["OA"] = (TP + TN) / total if total > 0 else 0.0

#     po = metrics["OA"]
#     pe = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / (total ** 2) if total > 0 else 0.0
#     metrics["Kappa"] = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

#     metrics["Precision"] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
#     metrics["Recall"] = TP / (TP + FN) if (TP + FN) > 0 else 0.0

#     precision = metrics["Precision"]
#     recall = metrics["Recall"]

#     metrics["F1"] = (
#         2 * precision * recall / (precision + recall)
#         if (precision + recall) > 0
#         else 0.0
#     )

#     metrics["IoU"] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

#     metrics["confusion_matrix"] = {
#         "TP": int(TP),
#         "TN": int(TN),
#         "FP": int(FP),
#         "FN": int(FN),
#     }

#     return metrics


# def create_visualization(pred_binary, target, img_shape):
#     """
#     创建指定颜色规则的可视化图像

#     Args:
#         pred_binary: numpy array, shape [H, W], 值为 0/1
#         target: numpy array, shape [H, W], 值为 0/1
#         img_shape: tuple, (H, W)

#     Returns:
#         vis_img: numpy array, shape [H, W, 3], BGR
#     """

#     vis_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

#     # TP：真实变化，预测变化，白色
#     mask_correct_change = (target == 1) & (pred_binary == 1)
#     vis_img[mask_correct_change] = COLOR_MAP["correct_change"]

#     # TN：真实未变化，预测未变化，黑色
#     mask_correct_no_change = (target == 0) & (pred_binary == 0)
#     vis_img[mask_correct_no_change] = COLOR_MAP["correct_no_change"]

#     # FP：真实未变化，预测变化，绿色
#     mask_false_change = (target == 0) & (pred_binary == 1)
#     vis_img[mask_false_change] = COLOR_MAP["false_predicted_change"]

#     # FN：真实变化，预测未变化，红色
#     mask_unpredicted_change = (target == 1) & (pred_binary == 0)
#     vis_img[mask_unpredicted_change] = COLOR_MAP["unpredicted_change"]

#     return vis_img


# @torch.no_grad()
# def test_and_visualize(args, model, test_loader, test_data, device):
#     """执行测试并生成可视化结果"""

#     print("=" * 60)
#     print("Starting Test and Visualization")
#     print("=" * 60)

#     vis_dir = osp(args.save_dir, "visualization")
#     os.makedirs(vis_dir, exist_ok=True)

#     all_preds = []
#     all_targets = []

#     model.eval()

#     for batch_idx, batched_inputs in enumerate(test_loader):
#         img, target = batched_inputs

#         pre_img = img[:, 0:3].to(device, non_blocking=True).float()
#         post_img = img[:, 3:6].to(device, non_blocking=True).float()

#         target_np = target.cpu().numpy().astype(np.uint8)

#         output = model(pre_img, post_img)

#         if isinstance(output, (tuple, list)):
#             pred = output[0]
#         else:
#             pred = output

#         pred_binary = torch.where(
#             pred > 0.5,
#             torch.ones_like(pred),
#             torch.zeros_like(pred)
#         )

#         pred_binary = pred_binary.cpu().numpy().astype(np.uint8)

#         all_preds.append(pred_binary)
#         all_targets.append(target_np)

#         for sample_idx in range(pred_binary.shape[0]):
#             global_idx = batch_idx * args.batch_size + sample_idx

#             if global_idx >= len(test_data):
#                 break

#             # 兼容 [B, 1, H, W] 或 [B, H, W]
#             if pred_binary.ndim == 4:
#                 pred_sample = pred_binary[sample_idx, 0]
#             else:
#                 pred_sample = pred_binary[sample_idx]

#             if target_np.ndim == 4:
#                 target_sample = target_np[sample_idx, 0]
#             else:
#                 target_sample = target_np[sample_idx]

#             vis_img = create_visualization(
#                 pred_sample,
#                 target_sample,
#                 pred_sample.shape
#             )

#             vis_path = osp(vis_dir, f"test_{global_idx:04d}_vis.png")
#             cv2.imwrite(vis_path, vis_img)

#             print(f"✅ Saved visualization for sample {global_idx} to {vis_path}")

#     all_preds = np.concatenate(all_preds, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)

#     # 兼容 [B, 1, H, W] 或 [B, H, W]
#     if all_preds.ndim == 4:
#         all_preds_for_metric = all_preds[:, 0]
#     else:
#         all_preds_for_metric = all_preds

#     if all_targets.ndim == 4:
#         all_targets_for_metric = all_targets[:, 0]
#     else:
#         all_targets_for_metric = all_targets

#     overall_metrics = calculate_metrics(
#         all_preds_for_metric,
#         all_targets_for_metric
#     )

#     print("\n" + "=" * 60)
#     print("📊 Final Test Metrics")
#     print("=" * 60)
#     print(f"Overall Accuracy (OA):    {overall_metrics['OA']:.4f}")
#     print(f"Kappa Coefficient:        {overall_metrics['Kappa']:.4f}")
#     print(f"Precision:                {overall_metrics['Precision']:.4f}")
#     print(f"Recall:                   {overall_metrics['Recall']:.4f}")
#     print(f"F1 Score:                 {overall_metrics['F1']:.4f}")
#     print(f"IoU:                      {overall_metrics['IoU']:.4f}")
#     print("-" * 60)
#     print(f"TP:                       {overall_metrics['confusion_matrix']['TP']}")
#     print(f"TN:                       {overall_metrics['confusion_matrix']['TN']}")
#     print(f"FP:                       {overall_metrics['confusion_matrix']['FP']}")
#     print(f"FN:                       {overall_metrics['confusion_matrix']['FN']}")
#     print("=" * 60)

#     metrics_path = osp(args.save_dir, "test_metrics.txt")

#     with open(metrics_path, "w", encoding="utf-8") as f:
#         f.write("Test Set Evaluation Metrics\n")
#         f.write("=" * 50 + "\n")
#         f.write(f"OA:        {overall_metrics['OA']:.4f}\n")
#         f.write(f"Kappa:     {overall_metrics['Kappa']:.4f}\n")
#         f.write(f"Precision: {overall_metrics['Precision']:.4f}\n")
#         f.write(f"Recall:    {overall_metrics['Recall']:.4f}\n")
#         f.write(f"F1:        {overall_metrics['F1']:.4f}\n")
#         f.write(f"IoU:       {overall_metrics['IoU']:.4f}\n")
#         f.write("\nConfusion Matrix:\n")
#         f.write(f"TP: {overall_metrics['confusion_matrix']['TP']}\n")
#         f.write(f"TN: {overall_metrics['confusion_matrix']['TN']}\n")
#         f.write(f"FP: {overall_metrics['confusion_matrix']['FP']}\n")
#         f.write(f"FN: {overall_metrics['confusion_matrix']['FN']}\n")

#     print(f"\n✅ Metrics saved to {metrics_path}")
#     print(f"✅ Visualizations saved to {vis_dir}")

#     return overall_metrics


# def main():
#     """主函数"""

#     args = parse_args()

#     # 测试专用默认参数
#     # 注意：getattr 第二个参数必须是属性名，不是路径
#     args.weight_path = getattr(
#         args,
#         "weight_path",
#         "exp_unified/U3_contrastive_only_sysu/best_model.pth"
#     )

#     args.save_dir = getattr(
#         args,
#         "save_dir",
#         "eexp_unified/U3_contrastive_only_sysu/test_results"
#     )

#     args.batch_size = getattr(args, "batch_size", 1)
#     args.num_workers = getattr(args, "num_workers", 8)

#     if not hasattr(args, "gpu_id"):
#         args.gpu_id = 0

#     if args.weight_path is None:
#         raise ValueError("Please specify weight path with --weight_path")

#     os.makedirs(args.save_dir, exist_ok=True)

#     print_config(args)

#     set_env(args)

#     device = get_device()

#     print(f"Using device: {device}")

#     model = load_model(args, device)

#     # 计算 FLOPs 和参数量
#     calculate_flops_and_params(args, model, device)

#     # 创建测试集
#     test_loader, test_data = create_test_loader(args)

#     # 测试和可视化
#     test_and_visualize(args, model, test_loader, test_data, device)

#     print("\n🎉 Test, visualization, FLOPs and parameters calculation completed successfully!")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 & 可视化脚本（严格匹配 U3 权重）

功能：
1. 加载训练好的 U3 权重
2. 对测试集进行预测
3. 保存二值预测和可视化结果（BGR颜色规则）
4. 计算完整指标：OA, Kappa, Precision, Recall, F1, IoU
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from os.path import join as osp

# 添加项目路径
sys.path.insert(0, '.')

# 导入必要模块
from config_unified import get_ablation_configs, print_config
from data.dataset import BCDDataset
from data.transforms import BCDTransforms
from unified_trainer import UnifiedTrainer

# 颜色映射 (BGR)
COLOR_MAP = {
    'unpredicted_change': (0, 0, 255),    # 红色
    'false_predicted_change': (0, 255, 0),# 绿色
    'correct_change': (255, 255, 255),    # 白色
    'correct_no_change': (0, 0, 0)        # 黑色
}


def load_model(weight_path, args):
    """严格匹配 U3 配置，加载权重"""
    # 使用 U3 配置
    config_dict = get_ablation_configs()
    args = config_dict["U4"]  # 严格匹配训练结构

    model = UnifiedTrainer(args).cuda().float()
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight not found: {weight_path}")
    state = torch.load(weight_path, map_location='cuda')
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded U4 weights from {weight_path}")
    return model, args


def create_test_loader(args):
    """测试集 DataLoader"""
    _, val_transform = BCDTransforms.get_transform_pipelines(args)
    test_data = BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,  # 可视化建议 batch_size=1
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    return test_loader, test_data


def calculate_metrics(pred_binary, target):
    """计算 OA, Kappa, Precision, Recall, F1, IoU"""
    pred_flat = pred_binary.reshape(-1)
    target_flat = target.reshape(-1)

    TP = np.sum((pred_flat==1)&(target_flat==1))
    TN = np.sum((pred_flat==0)&(target_flat==0))
    FP = np.sum((pred_flat==1)&(target_flat==0))
    FN = np.sum((pred_flat==0)&(target_flat==1))

    total = TP+TN+FP+FN
    OA = (TP+TN)/total if total>0 else 0
    po = OA
    pe = ((TP+FP)*(TP+FN)+(TN+FN)*(TN+FP))/(total**2)
    Kappa = (po-pe)/(1-pe) if (1-pe)!=0 else 0
    Precision = TP/(TP+FP) if (TP+FP)>0 else 0
    Recall = TP/(TP+FN) if (TP+FN)>0 else 0
    F1 = 2*Precision*Recall/(Precision+Recall) if (Precision+Recall)>0 else 0
    IoU = TP/(TP+FP+FN) if (TP+FP+FN)>0 else 0

    return {"OA":OA,"Kappa":Kappa,"Precision":Precision,"Recall":Recall,"F1":F1,"IoU":IoU,
            "TP":TP,"TN":TN,"FP":FP,"FN":FN}


def create_visualization(pred_binary, target):
    """生成颜色可视化图"""
    H,W = pred_binary.shape
    vis = np.zeros((H,W,3),dtype=np.uint8)
    vis[(target==1)&(pred_binary==1)] = COLOR_MAP['correct_change']
    vis[(target==0)&(pred_binary==0)] = COLOR_MAP['correct_no_change']
    vis[(target==0)&(pred_binary==1)] = COLOR_MAP['false_predicted_change']
    vis[(target==1)&(pred_binary==0)] = COLOR_MAP['unpredicted_change']
    return vis


@torch.no_grad()
def test_and_visualize(model, test_loader, test_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    all_preds=[]
    all_targets=[]
    for idx, (img, target) in enumerate(test_loader):
        pre_img = img[:,0:3].cuda().float()
        post_img = img[:,3:6].cuda().float()
        target_np = target.numpy()

        pred, _, _ = model(pre_img, post_img)
        pred_binary = (pred>0.5).cpu().numpy().astype(np.uint8)[:,0]

        all_preds.append(pred_binary)
        all_targets.append(target_np[:,0])

        vis = create_visualization(pred_binary[0], target_np[0,0])
        cv2.imwrite(osp(save_dir,f"test_{idx:03d}_vis.png"), vis)
    all_preds = np.concatenate(all_preds,0)
    all_targets = np.concatenate(all_targets,0)
    metrics = calculate_metrics(all_preds, all_targets)
    print("\n📊 Test Metrics:")
    for k,v in metrics.items():
        print(f"{k}: {v}")
    print(f"✅ Visualizations saved to {save_dir}")


def main():
    weight_path = "exp_unified/U4_unified_full_sysu/best_model.pth"
    save_dir = "exp_unified/U4_unified_full_sysu/test_vis"

    model, args = load_model(weight_path, None)
    print_config(args)
    test_loader, test_data = create_test_loader(args)
    test_and_visualize(model, test_loader, test_data, save_dir)


if __name__ == "__main__":
    main()
