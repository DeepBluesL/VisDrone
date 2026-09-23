# VisDrone: migrating occluded-MNIST modules to object detection

将原 MNIST 遮挡比较模型中的实际使用模块迁移到 **VisDrone2019-DET 十类目标检测**。
基于 YOLOv8n 检测框架，比较 C3k2、C3Ghost、SE、EnhancedSPPF 和修正后的 Gabor
滤波器，提供单模块、完整组合、逐项移除及随机滤波器对照。

本项目迁移特征提取模块，使用检测框架的定位、分类和 DFL 损失。
MNIST 的数字比较头、全局池化、左右交换标签和特征范数排序损失不适用于该检测任务。

## 已完成的实验

**13/13 组已实际训练完成**：固定 2,048 张训练图、全部 548 张验证图，每组 10 epochs，
输入 512，batch 16，seed 179，从零训练。全部 8,629 张 train/val/test-dev 原始图片已下载，
test-dev 未参与训练或模型选择。[下载与校验记录](results/data/source_provenance.json)

| 配置 | mAP50 (%) | mAP50–95 (%) | 参数量 |
| --- | ---: | ---: | ---: |
| 基线 YOLOv8n | 9.64 | 4.47 | 3,012,798 |
| Gabor 单模块 | 10.30 | 4.87 | 3,012,878 |
| 同结构固定随机滤波器 | **10.49** | **4.99** | 3,012,878 |
| 五模块完整组合 | 8.76 | 4.04 | 2,497,232 |

本轮最高分来自随机滤波器对照，不能将 Gabor 相对基线的提升解释为 Gabor 先验的独特收益。
完整组合参数量减少约 17.1%，但 mAP50–95 低于基线约 0.43 个百分点。
这些是短程单种子的开发集观测，未证明充分训练后的排序或统计显著性。

[完整结果与消融报告](results/pilot/REPORT.md) · [指标 CSV](results/pilot/summary.csv) ·
[实验发现](docs/FINDINGS.md) · [13 组权重](checkpoints/pilot) ·
[权重及记录独立复核](results/pilot/verification.json) ·
[测试记录](results/pilot/validation.json) · [发布产物 SHA-256 清单](results/artifact_manifest.json)

![基线与五种单模块的真实精度对比](assets/pilot/comparison.png)

![完整组合、逐项移除和随机滤波器对照](assets/pilot/ablation.png)

![自然遮挡分组召回率](assets/pilot/occlusion_recall.png)

![同一组验证图片上的基线、完整组合和最佳配置检测结果](assets/pilot/predictions.png)

## 实验矩阵

| 分组 | 配置 |
| --- | --- |
| 基线 | YOLOv8n，十类输出，从零训练 |
| 单模块 × 5 | C3k2 / Ghost / SE / EnhancedSPPF / Gabor |
| 全部组合 | 五种模块同时使用 |
| 消融 × 5 | 全部组合中逐个移除一种模块 |
| 滤波器对照 | 与 Gabor 完全相同结构的固定随机滤波器 |

每组使用相同训练图片、验证图片、优化器、图像尺寸、训练轮数和随机种子。
相同位置的迁移模块使用独立且一致的初始化种子；未改动的检测头具有相同初始权重。
原始代码的 LeakyReLU 和局部 BatchNorm 默认设置保留，因此实验比较的是整个模块实现。

## 运行

建议 Python 3.10，先安装适配 GPU 的 PyTorch / torchvision，再安装：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/download_data.py
python scripts/prepare_data.py --train-limit 2048 --seed 179
python scripts/run_suite.py --suite pilot_reproduction --epochs 10 --imgsz 512 --batch 16 --seeds 179 --expected-train 2048 --expected-val 548
python scripts/report.py --suite pilot_reproduction
```

发布的结果保存在 `pilot`；复现实验使用新的 `pilot_reproduction` 名称，避免覆盖已归档结果。
运行路径和依赖版本也参与协议核验，因此不同机器上的新实验应使用独立 suite。

复核发布权重，并重建定性图和遮挡诊断：

```bash
python scripts/verify_results.py --suite pilot --expected-runs 13
python scripts/visualize_predictions.py --weights checkpoints/pilot/baseline_s179.pt checkpoints/pilot/full_s179.pt checkpoints/pilot/random_stem_s179.pt --labels baseline full random_stem --output assets/pilot/predictions.png
python scripts/evaluate_occlusion.py --weights checkpoints/pilot/baseline_s179.pt --output results/pilot/baseline_s179/occlusion.json
```

遮挡诊断对每一组权重重复同一命令，并使用相应结果目录；随后运行 `report.py` 重新汇总。
召回率阈值固定为置信度 0.05、匹配 IoU 0.5，图像选择与预测无关。

`--train-limit 0` 使用完整的 6,471 张训练图片；默认验证集始终为官方 548 张。
下载脚本还下载 1,610 张 test-dev 图片，但首轮实验不使用 test-dev 选择模型。
全部原始数据仅保存在 `data/`，不进入 Git。

更长训练与多种子实验（以下为配置示例，不代表已完成）：

```bash
python scripts/prepare_data.py --train-limit 0 --seed 179
python scripts/run_suite.py --suite full_100ep --epochs 100 --imgsz 640 --batch 16 --seeds 179 2026 3407 --expected-train 6471 --expected-val 548
python scripts/report.py --suite full_100ep
```

数据准备改变后应使用新的 suite 名称。训练和验证都通过命令行数据 YAML 指定。
`run_suite.py` 会核对上述预期图片数，并把实际 dataset YAML、数据准备清单、数据内容指纹、
训练代码指纹和依赖版本保存在 `results/<suite>/`；任何一项改变都会拒绝复用该 suite。
`runs/` 保存完整训练输出；`results/` 保存可提交的指标和逐轮 CSV；
`checkpoints/` 保存各组最佳验证权重；`assets/` 保存由真实指标生成的图。

## 解读限制

短程、单种子、从零训练的实验用于检查模块迁移与固定预算下的表现，不能证明收敛后的优劣。
主指标是 **YOLO 转换标注上的 mAP50–95**，范围为 0–1，图表显示百分数。
转换排除 ignored region、score=0 和非目标类别，但验证器没有实现官方 VisDrone 的忽略区域匹配，
因此这里的分数不能直接作为官方挑战榜单分数。验证集同时用于挑选最佳 epoch，未声称独立测试成绩。

来源与许可见 [ATTRIBUTION](docs/ATTRIBUTION.md)。代码采用 AGPL-3.0；数据遵循原数据集条款。
