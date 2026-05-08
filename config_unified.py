# Copyright (c) Unified Framework Configuration
# All rights reserved.

"""
统一框架配置文件

包含所有训练、模型、数据相关的配置参数
支持命令行参数覆盖
"""

import argparse
from argparse import Namespace


def str2bool(v):
    """字符串转 bool，用于命令行参数"""
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True

    if v.lower() in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_base_config():
    """
    获取基础配置（所有实验通用）

    Returns:
        Namespace: 基础配置参数
    """

    config = Namespace(
        # ========== 实验标识 ==========
        exp_name="diff_attn_only_sysu",
        exp_id="U1",

        # ========== 数据集配置 ==========
        dataset="SYSU-CD",
        file_root="autodl-tmp/SYSU-CD",
        in_height=256,
        in_width=256,
        num_perception_frame=1,
        num_class=1,

        # ========== 训练配置 ==========
        max_steps=80000,
        max_epochs=None,
        batch_size=16,
        num_workers=8,

        # ========== 优化器配置 ==========
        optimizer="adamw",
        lr=5e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        max_grad_norm=1.0,

        # ========== 学习率调度器配置 ==========
        scheduler_type="warmup_poly",
        poly_power=0.9,
        warmup_steps=1000,
        warmup_updates=None,
        min_lr=1e-6,

        # ========== 差分注意力配置 ==========
        use_differential_attention=False,
        use_gating=False,
        use_channel_attention=False,
        use_spatial_attention=False,
        use_cross_level=False,
        use_top_down=False,
        use_bottom_up=False,

        # ========== 对比学习配置 ==========
        use_contrastive=True,
        embed_dim=256,
        temperature=0.07,
        contrast_weight=0.1,
        max_neg_samples=256,
        chunk_size=2048,
        scale_weights=[0.1, 0.4, 0.7, 1.0],

        # ========== 混合精度训练 ==========
        use_amp=False,
        gradient_accumulation_steps=1,
        use_gradient_checkpointing=True,

        # ========== 早停配置 ==========
        patience=50,
        min_delta=1e-4,

        # ========== 模型配置 ==========
        pretrained="autodl-tmp/Change3D-master/X3D_L.pyth",

        # ========== 保存配置 ==========
        save_dir="exp_unified/U1_diff_attn_only_sysu",
        log_file="train_log.txt",
        save_freq=1,
        val_freq=1,

        # ========== 可视化配置 ==========
        vis_freq=500,
        use_tensorboard=True,
        tensorboard_dir="runs/unified",

        # ========== 其他配置 ==========
        gpu_id=0,
        seed=16,
        resume=None,
        auto_resume=False,
        monitor_memory=True,
    )

    return config


def get_unified_config():
    """
    获取统一框架的默认配置（推荐）

    Returns:
        Namespace: 完整配置
    """

    config = get_base_config()

    config.exp_name = "contrastive_only_sysu"
    config.exp_id = "U3"

    # 启用完整模型
    config.use_differential_attention = True
    config.use_gating =True
    config.use_channel_attention = True
    config.use_spatial_attention = True

    # 你当前日志里 cross_level 是 False，这里保持一致
    config.use_cross_level = False
    config.use_top_down = False
    config.use_bottom_up = False

    # 启用对比学习
    config.use_contrastive = True
    config.contrast_weight = 0.1

    # 显存友好配置
    config.embed_dim = 256
    config.max_neg_samples = 256
    config.chunk_size = 4096
    config.use_amp = True
    config.batch_size = 16
    config.gradient_accumulation_steps = 1

    config.save_dir = f"./exp_unified_U3/{config.exp_id}_{config.exp_name}"

    return config


def get_ablation_configs():
    """
    获取消融实验的所有配置

    Returns:
        dict: 消融实验配置字典
    """

    configs = {}

    # ========== U0: Baseline ==========
    config_e0 = get_base_config()
    config_e0.exp_name = "baseline_sysu"
    config_e0.exp_id = "U0"
    config_e0.use_differential_attention = False
    config_e0.use_cross_level = False
    config_e0.use_contrastive = False
    config_e0.contrast_weight = 0.0
    config_e0.save_dir = f"./exp_unified/{config_e0.exp_id}_{config_e0.exp_name}"
    configs["U0"] = config_e0

    # ========== U1: 只用差分注意力 ==========
    config_e1 = get_base_config()
    config_e1.exp_name = "diff_attn_only_sysu"
    config_e1.exp_id = "U1"
    config_e1.use_differential_attention = True
    config_e1.use_cross_level = False
    config_e1.use_contrastive = False
    config_e1.contrast_weight = 0.0
    config_e1.save_dir = f"./exp_unified/{config_e1.exp_id}_{config_e1.exp_name}"
    configs["U1"] = config_e1

    # ========== U2: 差分注意力 + 跨层级 ==========
    config_e2 = get_base_config()
    config_e2.exp_name = "diff_attn_cross_level_sysu"
    config_e2.exp_id = "U2"
    config_e2.use_differential_attention = True
    config_e2.use_cross_level = True
    config_e2.use_top_down = True
    config_e2.use_bottom_up = True
    config_e2.use_contrastive = False
    config_e2.contrast_weight = 0.0
    config_e2.save_dir = f"./exp_unified/{config_e2.exp_id}_{config_e2.exp_name}"
    configs["U2"] = config_e2

    # ========== U3: 只用对比学习 ==========
    config_e3 = get_base_config()
    config_e3.exp_name = "contrastive_only_sysu"
    config_e3.exp_id = "U3"
    config_e3.use_differential_attention = False
    config_e3.use_cross_level = False
    config_e3.use_contrastive = True
    config_e3.contrast_weight = 0.1
    config_e3.save_dir = f"./exp_unified/{config_e3.exp_id}_{config_e3.exp_name}"
    configs["U3"] = config_e3

    # ========== U4: 统一框架 ==========
    config_e4 = get_unified_config()
    config_e4.exp_name = "unified_full_sysu"
    config_e4.exp_id = "U4"
    config_e4.save_dir = f"./exp_unified/{config_e4.exp_id}_{config_e4.exp_name}"
    configs["U4"] = config_e4

    # ========== H1: 对比权重 0.05 ==========
    config_h1 = get_unified_config()
    config_h1.exp_name = "unified_cw0.05_sysu"
    config_h1.exp_id = "H1"
    config_h1.contrast_weight = 0.05
    config_h1.save_dir = f"./exp_unified/{config_h1.exp_id}_{config_h1.exp_name}"
    configs["H1"] = config_h1

    # ========== H2: 对比权重 0.2 ==========
    config_h2 = get_unified_config()
    config_h2.exp_name = "unified_cw0.2_sysu"
    config_h2.exp_id = "H2"
    config_h2.contrast_weight = 0.2
    config_h2.save_dir = f"./exp_unified/{config_h2.exp_id}_{config_h2.exp_name}"
    configs["H2"] = config_h2

    # ========== H3: 对比权重 0.3 ==========
    config_h3 = get_unified_config()
    config_h3.exp_name = "unified_cw0.3_sysu"
    config_h3.exp_id = "H3"
    config_h3.contrast_weight = 0.3
    config_h3.save_dir = f"./exp_unified/{config_h3.exp_id}_{config_h3.exp_name}"
    configs["H3"] = config_h3

    return configs


def get_dataset_specific_config(dataset_name):
    """
    获取特定数据集的配置

    Args:
        dataset_name: 数据集名称

    Returns:
        Namespace: 数据集特定配置
    """

    config = get_unified_config()

    if dataset_name == "WHU-CD":
        config.dataset = "WHU-CD"
        config.file_root = "/path/to/WHU-CD-256"
        config.in_height = 256
        config.in_width = 256
        config.batch_size = 8
        config.gradient_accumulation_steps = 2
        config.max_steps = 80000

    elif dataset_name == "LEVIR-CD":
        config.dataset = "LEVIR-CD"
        config.file_root = "/path/to/LEVIR-CD-256"
        config.in_height = 256
        config.in_width = 256
        config.batch_size = 8
        config.gradient_accumulation_steps = 2
        config.max_steps = 100000

    elif dataset_name in ["SYSU", "SYSU-CD"]:
        config.dataset = "SYSU-CD"
        config.file_root = "autodl-tmp/SYSU-CD"
        config.in_height = 256
        config.in_width = 256
        config.batch_size = 4
        config.gradient_accumulation_steps = 4
        config.max_steps = 80000
        config.use_amp = True
        config.embed_dim = 128
        config.max_neg_samples = 128
        config.chunk_size = 2048

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return config


def parse_args():
    """
    解析命令行参数

    Returns:
        Namespace: 合并后的配置
    """

    parser = argparse.ArgumentParser(
        description="Unified Framework Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== 基础参数 ==========
    parser.add_argument(
        "--config",
        type=str,
        default="unified",
        choices=["base", "unified", "U0", "U1", "U2", "U3", "U4", "H1", "H2", "H3"],
        help="Configuration preset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["WHU-CD", "LEVIR-CD", "CLCD", "SYSU-CD", "SYSU"],
        help="Dataset name"
    )

    parser.add_argument(
        "--file_root",
        type=str,
        default=None,
        help="Dataset root directory"
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name"
    )

    parser.add_argument(
        "--exp_id",
        type=str,
        default=None,
        help="Experiment ID"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints and logs"
    )

    # ========== 训练参数 ==========
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum epochs")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of dataloader workers")

    # ========== 优化器参数 ==========
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer type")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max grad norm")

    # ========== 学习率调度参数 ==========
    parser.add_argument("--scheduler_type", type=str, default=None, help="Scheduler type")
    parser.add_argument("--poly_power", type=float, default=None, help="Poly scheduler power")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--warmup_updates", type=int, default=None, help="Warmup updates")
    parser.add_argument("--min_lr", type=float, default=None, help="Minimum learning rate")

    # ========== 差分注意力参数 ==========
    parser.add_argument(
        "--use_differential_attention",
        type=str2bool,
        default=None,
        help="Use differential attention"
    )

    parser.add_argument(
        "--use_gating",
        type=str2bool,
        default=None,
        help="Use gating"
    )

    parser.add_argument(
        "--use_channel_attention",
        type=str2bool,
        default=None,
        help="Use channel attention"
    )

    parser.add_argument(
        "--use_spatial_attention",
        type=str2bool,
        default=None,
        help="Use spatial attention"
    )

    parser.add_argument(
        "--use_cross_level",
        type=str2bool,
        default=None,
        help="Use cross-level propagation"
    )

    parser.add_argument(
        "--use_top_down",
        type=str2bool,
        default=None,
        help="Use top-down propagation"
    )

    parser.add_argument(
        "--use_bottom_up",
        type=str2bool,
        default=None,
        help="Use bottom-up propagation"
    )

    # ========== 对比学习参数 ==========
    parser.add_argument(
        "--use_contrastive",
        type=str2bool,
        default=None,
        help="Use contrastive learning"
    )

    parser.add_argument("--contrast_weight", type=float, default=None, help="Contrastive loss weight")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for contrastive loss")
    parser.add_argument("--embed_dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--max_neg_samples", type=int, default=None, help="Maximum negative samples")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size for contrastive computation")

    # ========== 显存优化参数 ==========
    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=None,
        help="Use automatic mixed precision"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--use_gradient_checkpointing",
        type=str2bool,
        default=None,
        help="Use gradient checkpointing"
    )

    # ========== 早停参数 ==========
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=None, help="Early stopping min delta")

    # ========== 模型参数 ==========
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained weight path")

    # ========== 其他参数 ==========
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU ID")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    parser.add_argument(
        "--auto_resume",
        type=str2bool,
        default=None,
        help="Auto resume from checkpoint.pth.tar"
    )

    parser.add_argument(
        "--monitor_memory",
        type=str2bool,
        default=None,
        help="Monitor GPU memory"
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="exp_unified/U3_contrastive_only_sysu/best_model.pth",
        help="Path to trained model weights"
)
    args = parser.parse_args()

    # 根据配置选择基础配置
    if args.config in ["U0", "U1", "U2", "U3", "U4", "H1", "H2", "H3"]:
        configs = get_ablation_configs()
        config = configs[args.config]

    elif args.config == "unified":
        config = get_unified_config()

    else:
        config = get_base_config()

    # 命令行参数覆盖
    for key, value in vars(args).items():
        if key == "config":
            continue

        if value is not None:
            if key == "dataset" and value == "SYSU":
                value = "SYSU-CD"

            setattr(config, key, value)

    # 如果 save_dir 没有手动传入，则根据 exp_id 和 exp_name 自动更新
    if args.save_dir is None:
        if config.exp_id:
            if config.exp_id in ["U0", "U1", "U2", "U3", "U4", "H1", "H2", "H3"]:
                config.save_dir = f"./exp_unified/{config.exp_id}_{config.exp_name}"
            else:
                config.save_dir = f"./exp_unified_base/{config.exp_id}_{config.exp_name}"
        else:
            config.save_dir = f"./exp_unified_base/{config.exp_name}"

    validate_config(config)

    return config


def print_config(config):
    """
    打印配置信息

    Args:
        config: 配置对象
    """

    print("\n" + "=" * 80)
    print(f"Configuration: {config.exp_id} - {config.exp_name}")
    print("=" * 80)

    print("\n[Dataset]")
    print(f"  Name: {config.dataset}")
    print(f"  Root: {config.file_root}")
    print(f"  Image Size: {config.in_height}x{config.in_width}")

    print("\n[Training]")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  Max Epochs: {config.max_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Scheduler: {config.scheduler_type}")

    print("\n[Differential Attention]")
    print(f"  Enabled: {config.use_differential_attention}")
    if config.use_differential_attention:
        print(f"  ├─ Gating: {config.use_gating}")
        print(f"  ├─ Channel Attention: {config.use_channel_attention}")
        print(f"  ├─ Spatial Attention: {config.use_spatial_attention}")
        print(f"  └─ Cross Level: {config.use_cross_level}")
        print(f"     ├─ Top Down: {config.use_top_down}")
        print(f"     └─ Bottom Up: {config.use_bottom_up}")

    print("\n[Contrastive Learning]")
    print(f"  Enabled: {config.use_contrastive}")
    if config.use_contrastive:
        print(f"  ├─ Weight: {config.contrast_weight}")
        print(f"  ├─ Embed Dim: {config.embed_dim}")
        print(f"  ├─ Temperature: {config.temperature}")
        print(f"  ├─ Max Neg Samples: {config.max_neg_samples}")
        print(f"  └─ Chunk Size: {config.chunk_size}")

    print("\n[Optimization]")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  AMP: {config.use_amp}")
    print(f"  Gradient Checkpointing: {config.use_gradient_checkpointing}")
    print(f"  Max Grad Norm: {config.max_grad_norm}")

    print("\n[Output]")
    print(f"  Save Dir: {config.save_dir}")
    print(f"  Log File: {config.log_file}")
    print(f"  Resume: {config.resume if config.resume else 'None'}")
    print(f"  Auto Resume: {config.auto_resume}")

    print("\n[Hardware]")
    print(f"  GPU ID: {config.gpu_id}")
    print(f"  Num Workers: {config.num_workers}")
    print(f"  Monitor Memory: {config.monitor_memory}")

    print("=" * 80 + "\n")


def validate_config(config):
    """
    验证配置的有效性

    Args:
        config: 配置对象

    Raises:
        ValueError: 配置无效时
    """

    # 检查数据集
    if config.dataset == "SYSU":
        config.dataset = "SYSU-CD"

    valid_datasets = ["WHU-CD", "LEVIR-CD", "CLCD", "SYSU-CD"]
    if config.dataset not in valid_datasets:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # 检查参数范围
    if config.use_contrastive:
        if not 0 < config.contrast_weight <= 1.0:
            raise ValueError(
                f"contrast_weight must be in (0, 1] when use_contrastive=True, "
                f"got {config.contrast_weight}"
            )
    else:
        if config.contrast_weight > 0:
            print("⚠️  Warning: use_contrastive=False but contrast_weight>0")
            print("   Setting contrast_weight=0")
            config.contrast_weight = 0.0

    if not 0 < config.temperature <= 1.0:
        raise ValueError(f"temperature must be in (0, 1], got {config.temperature}")

    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")

    if config.gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, "
            f"got {config.gradient_accumulation_steps}"
        )

    if config.embed_dim < 1:
        raise ValueError(f"embed_dim must be >= 1, got {config.embed_dim}")

    if config.max_neg_samples < 1:
        raise ValueError(f"max_neg_samples must be >= 1, got {config.max_neg_samples}")

    if config.chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {config.chunk_size}")

    # 仅提示，不强制报错
    if config.file_root in ["/path/to/WHU-CD-256", "/path/to/LEVIR-CD-256", "/path/to/SYSU-CD"]:
        print("⚠️  Warning: file_root is still a placeholder path.")
        print("   Please update it to your actual dataset path.")

    if config.pretrained == "pretrained/X3D_L.pyth":
        print("⚠️  Warning: pretrained path is still the default placeholder.")
        print("   Please update it to your actual pretrained weights path.")

    print("✅ Configuration validated successfully!\n")


if __name__ == "__main__":
    config = parse_args()
    print_config(config)
