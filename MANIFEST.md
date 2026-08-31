# Open-set / OOD 3D Occupancy 调研材料清单

下载和审计日期：2026-08-27

本目录收集了本次调研的 10 篇论文及其官方公开仓库。Git 仓库均为 shallow clone；下面记录的 commit 可以用于之后确认代码是否发生变化。

当前讨论收敛后的系统设计见：

- [`FINAL_CAMERA_LIDAR_OOD_OCCUPANCY_DESIGN.md`](FINAL_CAMERA_LIDAR_OOD_OCCUPANCY_DESIGN.md)：EFFOcc-style LiDAR-depth-guided Camera lifting、Camera/LiDAR BEV concat fusion、ground height、binary non-ground C2H、known-only detection、observability 和下游 2D collision BEV 的完整实现规格。

## 目录结构

```text
open_set_occupancy_survey/
├── REPORT.md       # 深入调研、代码审计和推荐方案
├── MANIFEST.md     # 本文件
├── papers/         # 官方 PDF
├── paper_text/     # pdftotext 生成的可检索文本
└── repos/          # 官方公开仓库
```

## 论文和代码

| 工作 | 论文 | 本地 PDF | 官方代码/资产 | 本地目录 | 下载 commit |
|---|---|---|---|---|---|
| ProOOD | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_ProOOD_Prototype-Guided_Out-of-Distribution_3D_Occupancy_Prediction_CVPR_2026_paper.pdf) | `papers/ProOOD.pdf` | [GitHub](https://github.com/7uHeng/ProOOD) | `repos/ProOOD` | `34c72d517a89` |
| LIDO | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Mosco_Learning_to_Identify_Out-of-Distribution_Objects_for_3D_LiDAR_Anomaly_Segmentation_CVPR_2026_paper.pdf) | `papers/LIDO.pdf` | [GitHub](https://github.com/SiMoM0/LIDO) | `repos/LIDO` | `bfc2113d1bb6` |
| OccOoD | [arXiv](https://arxiv.org/pdf/2506.21185) | `papers/OccOoD.pdf` | [GitHub](https://github.com/7uHeng/OccOoD) | `repos/OccOoD` | `ccd00a064de2` |
| LiPSOW / OWL | [IJCV/arXiv](https://arxiv.org/pdf/2409.14273) | `papers/LiPSOW.pdf` | [GitHub](https://github.com/g-meghana-reddy/open-world-panoptic-segmentation) | `repos/LiPSOW` | `87d76b398338` |
| ULOPS | [IROS 2025/arXiv](https://arxiv.org/pdf/2506.13265) | `papers/ULOPS.pdf` | [GitHub](https://github.com/robot-learning-freiburg/ulops) | `repos/ULOPS` | `80af50b038f5` |
| DOODLE | [WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Oh_DOODLE_Diffusion-based_Out-of-Distribution_Learning_for_Open-set_LiDAR_Semantic_Segmentation_WACV_2026_paper.pdf) | `papers/DOODLE.pdf` | [GitHub](https://github.com/chang9711/DOODLE) | `repos/DOODLE` | `cd48c652ec77` |
| EvOcc | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf) | `papers/EvOcc.pdf` | [GitHub](https://github.com/boschresearch/evocc) | `repos/EvOcc` | `e63550f5abe7` |
| STU | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Nekrasov_Spotting_the_Unexpected_STU_A_3D_LiDAR_Dataset_for_Anomaly_CVPR_2025_paper.pdf) | `papers/STU.pdf` | [GitHub](https://github.com/kumuji/stu_dataset) | `repos/STU` | `8f0f09c2ca4b` |
| CODA | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980399.pdf) | `papers/CODA.pdf` | [官方数据站/evaluator](https://github.com/coda-dataset/coda-dataset.github.io) | `repos/CODA` | `9c331cb5f696` |
| POP-3D | [NeurIPS 2023](https://papers.nips.cc/paper/2023/file/9e30acdeff572463c1db9b7de59de64c-Paper-Conference.pdf) | `papers/POP3D.pdf` | [GitHub](https://github.com/vobecant/POP3D) | `repos/POP3D` | `032ec26d4ed0` |

## 仓库实际内容快照

| 仓库 | Python 文件数 | 非 `.git` 文件数 | 算法实现状态 |
|---|---:|---:|---|
| ProOOD | 94 | 147 | SGN 版本较完整；VoxDet 分支 WIP |
| LIDO | 17 | 32 | 核心训练、推理、评测完整 |
| OccOoD | 93 | 198 | 核心模型完整，但数据生成和配置存在缺口 |
| LiPSOW | 105 | 191 | 两阶段核心代码存在，端到端说明不完整 |
| ULOPS | 646 | 756 | 基本是 MMDetection3D 底座；论文新增模块未公开 |
| DOODLE | 0 | 1 | 只有一行 README，无算法代码 |
| EvOcc | 0 | 3 | README 明确写着 code will be available soon |
| STU | 35 | 67 | 数据 evaluator、baseline 训练和 checkpoint 链接完整 |
| CODA | 2 | 231 | 官方网站、标注说明和 COCO evaluator；不是模型仓库 |
| POP3D | 375 | 413 | 模型、预处理、训练和评测代码基本完整 |

## 下载范围说明

- 已下载：10 篇论文 PDF、对应官方公开代码/数据站仓库、PDF 可检索文本。
- 未下载：大体积原始数据集和 pretrained checkpoints。本次任务要求的是论文与代码；权重和数据下载链接均保留在各仓库 README 中。
- 本次代码一致性结论来自论文逐段阅读、源码静态审计和 Python 语法解析，不等同于在完整数据集上重新训练并复现论文数值。
