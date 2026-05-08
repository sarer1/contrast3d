# Contrast3D

Unified Framework for Binary Change Detection with Differential Attention and Contrastive Learning.

---

## Training

```bash
python train_unified.py --config unified
```

---

## Ablation Experiments

```bash
python train_unified.py --config U0
python train_unified.py --config U1
python train_unified.py --config U2
python train_unified.py --config U3
python train_unified.py --config U4
```

---

## Test

```bash
python test_visualize_flops.py
```

---

<!-- ================= 中文说明 ================= -->

<!--
项目简介：

Contrast3D 是一个用于二值变化检测（BCD）的统一框架，
融合了：

1. Differential Attention
2. Contrastive Learning
3. Cross-Level Propagation
4. Pixel-Anchor Contrastive Loss
5. BANE Sampling
6. X3D Backbone

=================================================
数据集支持：

- SYSU-CD
- LEVIR-CD
- WHU-CD

=================================================
基础训练：

python train_unified.py --config unified

=================================================
消融实验：

U0：Baseline
U1：仅差分注意力
U2：差分注意力 + Cross-Level
U3：仅对比学习
U4：完整模型

=================================================
测试：

python test_visualize_flops.py

=================================================
指标：

- OA
- Kappa
- Precision
- Recall
- F1-score
- IoU

=================================================
-->
