# VisDrone2019-DET 迁移实验协议

## 研究问题

本实验把原遮挡 MNIST 项目中的五类结构干预迁移到 VisDrone2019-DET 十类目标检测。目标是比较这些干预在同一 YOLOv8n 检测骨架、同一数据、同一训练预算下的早期表现和计算代价。实验不使用预训练权重。

十个类别按 VisDrone 源标注 1–10 映射为 YOLO 0–9：pedestrian、people、bicycle、car、van、truck、tricycle、awning-tricycle、bus、motor。

## 数据与评估边界

数据准备脚本只使用官方 train 和 val 划分，不重新随机切分，也不让训练图像进入验证集。脚本默认使用完整训练集与 548 张官方验证图；当前 `pilot` 协议使用固定 `--seed 179` 选出的 2048 张训练图，并保留全部 548 张验证图。其他资源受限的预实验也可用 `--train-limit` 选择可复现的训练子集。除非明确设置 `--val-limit`，验证集保持完整。最终报告以数据准备清单中的实际数量为准。

VisDrone 原标注格式为 `x,y,w,h,score,class,truncation,occlusion`。转换时保留类别 1–10，将其映射到 YOLO 0–9；丢弃 `score=0`、类别 0 的忽略区域和类别 11 的 others；越界框裁剪到图像范围，裁剪后无正面积的框丢弃。转换统计、准确文件列表和 SHA-256 写入 `data/preparation_manifest.json`，源图像和源标注不修改。

这里报告的是转换后 YOLO 标签上的 Ultralytics AP，包括 AP50 和 AP50–95。官方 VisDrone DET 评测器还会在匹配阶段特殊处理忽略区域，普通 YOLO 标签不能表达这套规则。因此本项目结果应称为“YOLO 转换评估”，不能声称是官方 VisDrone DET 榜单成绩，也不应与使用官方评测器的论文数字直接比较。

## 模型与干预位置

所有变体共享 Ultralytics YOLOv8n 的检测头、特征融合路径、类别数和训练代码。只在以下固定骨干位置替换或包裹模块，索引对应构建后的 YOLOv8n `model` 顺序：

| 因素 | 固定位置 | 实现 |
|---|---:|---|
| Gabor | 0 | 用 32 个固定、零均值、单位 L2 范数的 5×5 Gabor 滤波器和可训练 1×1 投影替换 stride-2 stem |
| C3k2 | 2 | 用迁移的 C3k2 块替换原块 |
| SE | 2 | 在位置 2 的输出后增加 SE 通道注意力；与 C3k2 同时启用时包裹替换后的 C3k2 |
| Ghost | 4、6、8 | 在三个固定阶段用 C3Ghost 替换原块 |
| EnhancedSPPF | 9 | 用并行 3×3、5×5、7×7 最大池化及残差池化分支替换 SPPF |

Gabor 滤波器存为 buffer，不参与优化器更新。`random_stem` 使用相同数量、尺寸、归一化和 stride 的固定随机滤波器，只改变滤波器内容，用于区分 Gabor 先验与“固定滤波器 stem”本身的效果。模块初始化以运行种子和固定位置派生的局部种子控制，使单模块、完整模型和留一消融中同一位置的干预具有可比初始化。

迁移模块使用的自定义 Conv 提供与 Ultralytics 基线卷积一致的 Conv-BN fuse 推理路径，确保各变体在验证和推理优化下采用匹配的融合方式。

## 变体与比较规则

预定义变体包括：

- `baseline`：原 YOLOv8n；
- `c3k2`、`ghost`、`se`、`sppf`、`gabor`：分别只加入一个因素；
- `full`：同时加入五个因素；
- `without_c3k2`、`without_ghost`、`without_se`、`without_sppf`、`without_gabor`：从 `full` 中分别移除一个因素；
- `random_stem`：单独的固定随机滤波器对照。

单因素效果只以 `baseline` 为参照。留一消融只以 `full` 为参照。`random_stem` 只与 `baseline` 和单独的 `gabor` 比较，不并入五因素留一消融。这样可避免把不同参照下的差值解释成同一种效应。

## 固定训练配置

同一 suite 内所有变体和种子必须使用同一协议；`scripts/run_suite.py` 会把实际参数写入 `results/<suite>/protocol.json`，已有 suite 的协议发生变化时拒绝继续。当前固定项为：

| 项目 | 配置 |
|---|---|
| 初始化 | 从随机初始化训练，`pretrained=False` |
| 输入尺寸 | 512×512 |
| 默认预算 | 10 epochs |
| batch / workers | 16 / 4 |
| 有效 batch | `nbs=batch=16`，不做梯度累积 |
| 默认种子 | 179；多种子实验显式传入种子列表 |
| 优化器 | AdamW |
| 学习率 | `lr0=0.001`，`lrf=0.1`，余弦调度 |
| 正则与动量 | `weight_decay=0.0005`，`momentum=0.9` |
| warmup | `min(1.0, epochs/10)` epochs |
| checkpoint | 固定预算内验证集 AP50–95 最佳 checkpoint |
| 数值设置 | deterministic，AMP 关闭 |
| 几何增强 | translate 0.1、scale 0.3、水平翻转 0.5；degrees/shear/perspective/flipud 均为 0 |
| 颜色增强 | HSV h/s/v 为 0.015/0.7/0.4 |
| 组合增强 | mosaic、mixup、copy-paste 均关闭 |
| 其他 | cache 关闭，patience 0，max_det 500 |

`metrics.json` 记录实际 epoch、图像尺寸、batch、训练参数、环境版本、参数量、Conv/Linear GFLOPs、速度、数据 YAML 哈希、数据内容指纹、训练代码哈希、checkpoint 哈希和逐类 AP。GFLOPs 按 batch 1 的卷积与线性层 MACs×2 计算，包含固定 Gabor 卷积，不包含 BN、池化、激活、解码和 NMS。最终报告必须以这些运行记录为准，不用协议默认值替代缺失的实际结果。

## 汇总与不确定性

运行 `python scripts/report.py --suite pilot` 只读取 `results/pilot/*/metrics.json` 中 `status=completed` 的记录。汇总前必须确认所有运行的数据内容指纹、训练代码 SHA-256，以及 `environment_fingerprint` 中的 Python、torch、torchvision、Ultralytics、NumPy、Pillow 版本完全一致；缺失这些来源字段或固定训练配置不同都会拒绝汇总。最佳 checkpoint 所在的 `selected_epoch` 是结果，不是固定配置，允许因变体或种子而不同。报告只读取 `results/pilot/data_snapshot/preparation_manifest.json` 这一随 suite 保存的数据准备快照，不回退到当前 `data/preparation_manifest.json`，避免后来重新准备数据时改写历史实验解释。

报告中的 AP 统一换算为百分数并明确标注 `%`。同一变体有多个种子时报告均值和种子间样本标准差；只有一个种子时只报告单次观测，明确写为“未估计不确定性”，不显示虚假的零标准差。缺失变体、缺失 epoch 历史或缺失逐类数据不做插值和补值。

## 自然遮挡诊断

每个完成运行可以保存 `results/<suite>/<variant>_s<seed>/occlusion.json`。该诊断直接读取原始 VisDrone 验证标注的遮挡等级 0、1、2，不合成遮挡。合格 GT 与训练标签一致：`score>0`、源类别 1–10，边界框裁剪后仍有正面积。模型预测使用置信度阈值 0.05 和 NMS IoU 0.50；随后按置信度从高到低，与同类别 GT 做一对一 IoU≥0.50 贪心匹配，并在全局匹配完成后按遮挡等级分层。每层同时报告 recall 和 GT 总数。

这是固定工作点下的 class-aware ground-truth recall，不是 AP，也不复现官方 VisDrone 忽略区域匹配。诊断文件的权重 SHA-256 必须与对应运行 checkpoint 一致；不同文件列表、图像数量、subset 状态、阈值或 GT 数不能合并。若使用 `--max-images`，报告必须明确标为 subset smoke run。

诊断还可按目标大小分层。面积在裁剪后的原图像素坐标中计算：small `<32²`、medium `32²–<96²`、large `≥96²`。这些是分析分组，不代表模型输入缩放后的面积。

## 解释限制

十个 epoch 的从头训练主要用于验证迁移结构、训练稳定性和初步相对趋势。它不能证明任何变体已经收敛，也不能建立充分训练条件下的性能排序。单一种子的差异可能来自初始化、数据顺序或硬件数值噪声；需要多个预先确定的种子才能估计这种变异。

同一验证集既用于预算内最佳 checkpoint 选择，又用于最终数值报告，因此结果是开发集结果，不是独立测试集泛化估计。参数量和上述 Conv/Linear GFLOPs 描述模型复杂度，但不是完整端到端操作计数。`speed_ms.inference` 是 Ultralytics 验证器在批处理验证过程中记录的每图阶段时间，不是隔离环境中的单图端到端延迟；它还受 GPU、软件版本、batch 和测量实现影响，只能在记录的同一环境内比较。

结构干预同时可能改变参数量、计算量和优化难度。观察到 AP 差异时，应结合效率图、学习曲线和多种子方差解释，不能仅凭一次短训练把相关性表述为模块的确定因果收益。
