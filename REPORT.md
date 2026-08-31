# Open-set / OOD 3D Occupancy 深入调研与代码审计

日期：2026-08-27

## 结论先行

对于你的任务——预测道路场景的 3D occupancy，同时把沙发、轮胎、倒下的树干等训练类别之外的占据物识别为 `generic object`——我的结论是：

1. **最值得作为直接 occupancy 起点的是 ProOOD。** 它是这些工作中任务定义最贴近的一篇：直接在语义 occupancy voxel 上做 OOD detection，同时兼顾长尾类别。它不是开箱即用；关键的 PGTM 选择逻辑与论文公式不完全一致，使用前需要修正。
2. **最值得移植的 OOD 表征方法是 LIDO。** 它不依赖把所有未知物体压成一个可学习的“共同类别中心”，而是学习已知类别的支持区域，再用 prototype distance、entropy 和 objectosphere score 判断是否脱离已知分布。代码最紧凑，训练和推理入口也最清楚，但原任务是点级 LiDAR segmentation，需要改成 voxel 特征。
3. **最合理的最终系统不是照抄任何一篇，而是组合：EvOcc 式 GT + 你的 occupancy backbone + LIDO/ProOOD 式 novelty head + class-agnostic object grouping。** `generic object` 可以作为最终输出标签和训练辅助监督，但不应只是 semantic softmax 中唯一的第 K+1 类。
4. **Open vocabulary 应放在 unknown gate 之后。** POP-3D/CLIP 很适合尝试给未知物命名，但 text similarity 总会返回一个最相近名称，它本身不是可靠的“这个物体不属于任何已知类”的判据。
5. **当前不要以 ULOPS、DOODLE 或 EvOcc 的公开仓库为复现起点。** 三篇论文的思想都有价值，但当前公开代码分别是占位底座、单行 README、以及“代码即将公开”。

如果只能选一篇立刻开始：**选 ProOOD 做 occupancy 基线，参考 LIDO 重写 novelty scoring，不要原样复制 ProOOD 的 PGTM。**

---

## 1. 你的问题其实包含三个相互独立的变量

不要把下面三件事压进一个 softmax：

1. **几何状态**：voxel 是 `free`、`occupied`，还是 `unobserved`。
2. **已知语义**：若 voxel occupied，它属于 car、pedestrian、road、building 等哪个已知类。
3. **语义新颖性**：该 occupied voxel 是否落在所有已知类的分布之外。

其中 `unobserved` 是传感器可见性问题，`unknown/generic object` 是语义问题。一个 voxel 可以是：

- observed free；
- observed occupied + known class；
- observed occupied + generic/unknown object；
- unobserved，因此不能可靠判断 free 或 occupied。

推荐的条件分解是：

$$
P(\text{occupied}\mid x), \qquad
P(y=k\mid \text{occupied},x), \qquad
s_{\text{novel}}(x)\mid \text{occupied}.
$$

这也是为什么单纯加入一个 `unknown object` semantic 类不够：它只能学习训练中那批 unknown proxy 的判别边界，不能保证未来任意新类别都落入同一个特征簇。

---

## 2. 横向排名

分数为本次源码审计后的工程判断，5 分最好。“未知发现上限”指检测真正未见类别的潜力，不等于论文表格中的单个 benchmark 数值。

| 排名 | 方法 | 与你的 occupancy 任务匹配 | 未知发现上限 | 公开实现完整度 | 复现难度 | 论文—代码一致性 | 最适合的角色 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **ProOOD** | 5.0 | 4.0 | 4.0 | 中—高 | 3.0 | 直接 occupancy OOD 基线 |
| 2 | **LIDO** | 3.5 | 4.5 | 4.5 | 中 | 3.5 | 移植 novelty representation/scoring |
| 3 | **OccOoD** | 5.0 | 3.0 | 3.5 | 高 | 3.0 | VAA benchmark、CSSR/ASS 对照实验 |
| 4 | **LiPSOW / OWL** | 3.0 | 4.0 | 3.0 | 很高 | 4.0 | generic object 的实例级 grouping |
| 5 | **POP-3D** | 4.0 | 2.0（拒识）/ 5.0（命名） | 4.0 | 很高 | 4.0 | unknown gate 后的开放词汇命名 |
| 6 | **ULOPS** | 3.0 | 4.0（论文） | 0.5 | 当前不可复现 | 0.5 | 只借鉴 evidential losses 的思想 |
| 7 | **DOODLE** | 2.5 | 4.0（论文） | 0.0 | 当前不可复现 | 0.0 | 暂不采用，等待代码 |

### 两种不同的“最 promising”

- **现在最可能做成并产出可比较结果：ProOOD。** 原因是任务完全对齐、训练/评测入口和权重均已提供，而且额外参数只有论文报告的 0.28M。
- **从真正未见类别的建模原则看：LIDO 的 known-support learning 更干净。** 它学习“已知是什么”，而不是假设“所有未知长得像同一类”。其上限受特征质量和近分布 unknown 影响，但概念上比 K+1 softmax 更合适。
- **从物体命名的上限看：POP-3D 最高。** 但它解决的是“用文本查询这个 occupied voxel 像什么”，不是 calibrated OOD rejection，不能替代 unknown detector。

### 数据和评测资产单独排名

| 资产 | 直接价值 | 结论 |
|---|---|---|
| **STU** | 真实道路 LiDAR anomaly point/object benchmark | 最值得作为真实 OOD 测试集；不是 occupancy GT |
| **OccOoD VAA-KITTI / VAA-KITTI-360** | 合成 anomaly voxel benchmark | 与 occupancy 最直接，但生成代码没有公开，只能使用作者生成的数据 |
| **CODA** | 真实道路 corner cases，2D box | 适合图像侧 corner-case 检测/类别设计，不是 3D occupancy benchmark |
| **EvOcc** | free/occupied/unobserved 与证据式 GT 方法 | 对你的 GT 设计极重要，但当前无代码 |

---

## 3. 逐篇论文与代码核对

## 3.1 ProOOD：最贴题，但核心选择器需要修

本地材料：`papers/ProOOD.pdf`，`repos/ProOOD/`

### 论文做什么

ProOOD 直接研究 OOD 3D semantic occupancy，包含：

- **PGSI**：用 EMA class prototypes 补全被遮挡/未观测区域的语义特征；
- **PGTM**：用尾部类别 prototypes 从粗特征中挑选候选 voxel 并增强；
- **PBCL**：prototype-based contrastive loss；
- **EchoOOD**：融合 local logit coherence、local prototype distance、global prototype distance 的训练外 OOD score。

论文报告在 VAA-KITTI 上，EchoOOD 的 AuPRC-r 分别为 27.86、43.55、62.65（0.8/1.0/1.2 m），AUROC 64.31；增加约 0.28M 参数。它的优势不只是 anomaly score，还试图缓解 rare known classes 被误判成 unknown 的问题。

### 代码中确实存在的对应实现

- 两套 EMA prototype bank、prototype quality、warm-up、contrastive loss；
- auxiliary semantic occupancy head 和 masked-voxel prototype residual；
- EchoOOD 的三个 cue，按 scene 做归一化后取最大值；
- SemanticKITTI、KITTI-360、VAA-KITTI、VAA-KITTI-360、STU 配置；
- SGN 训练/评测脚本和四个 pretrained checkpoints。

核心文件：

- `repos/ProOOD/projects/mmdet3d_plugin/proood/prooodmodule/prototype.py`
- `repos/ProOOD/projects/mmdet3d_plugin/proood/prooodmodule/tailvoxelselector.py`
- `repos/ProOOD/projects/mmdet3d_plugin/proood/dense_heads/proood.py`

### 论文与代码不一致的关键点

1. **PGTM 的论文公式没有被完整实现。** 论文 Eq. 8 是先要求最大 tail similarity 大于 $\eta$、top-1/top-2 margin 大于 $\delta$，然后再 Top-K。代码计算了 margin `tail_certainty`，但选择时完全没有使用它，也没有 $\eta$/$\delta$ threshold，只按最大 similarity 排序取前 K。
2. **所谓 top-1 + top-2 的第二阶段实际上不可达。** `selected_top1 = top1_order[:k]` 后，`len(selected_top1) >= k` 对正常 k 恒成立，代码立即 `continue`，补充 top-2 的分支不会执行。
3. **maturity gating 比论文弱。** 论文描述的是按类别的质量、样本计数和 warm-up gate；代码先看全体已初始化 prototype 的比例和平均质量。EchoOOD 中虽然读出了 `prototype_initialized`，却没有用它过滤 global prototype scoring。
4. **工程假设较硬。** 多处配置和保存路径写死为 `/root/autodl-tmp/...`；head 内使用 `x3d[0]`、`img_metas[0]`，本质上依赖 batch size 1。
5. **只有 SGN 主分支完成。** README 将 VoxDet 分支标为 WIP。

### 判断

**方法思想与大部分模块一致，但不能把当前代码当成论文公式的精确参考实现。** 论文结果是否能由当前 commit 原样复现，需要数据和权重实跑才能确认；仅从源码看，PGTM 的消融结论不能直接归因于论文所写的“两阶段阈值 + margin + Top-K”。

复现评分：**3/5**。最适合你，但使用前应先修 PGTM、per-class maturity 和路径/batch 假设。

---

## 3.2 LIDO：最容易迁移的 OOD 核心

本地材料：`papers/LIDO.pdf`，`repos/LIDO/`

### 论文做什么

LIDO 是 point-level LiDAR anomaly segmentation：

- MinkUNet backbone；
- semantic head 从 true-positive features 构造置信度加权的 class prototypes；
- contrastive head 用 prototype contrastive loss 和 objectosphere loss 塑造已知类别特征空间；
- 推理时融合 prototype cosine distance × entropy 与 feature-norm score。

它的重要点是：**训练目标主要定义已知类的支持区域，不要求所有未来 unknown 共享一种语义。** 论文在 STU validation/test 上分别报告 AUROC 95.05/93.67，模型 21.7M 参数，在 A40 上约 38 ms、0.6 GB。

### 代码优点

- 17 个 Python 文件，结构很紧凑；
- install、train、infer、dataset、checkpoint 链接完整；
- prototype 累积、cosine loss、contrastive/objectosphere head 和最终融合 score 都能在源码中直接对应；
- 提供 Python 3.10、PyTorch 2.4/CUDA 12.4 的容器定义，比其他老 MMDetection3D 工程更容易搭环境。

### 不一致与风险

1. **objectosphere 半径不一致。** 论文 implementation details 使用 $r=5$；训练代码实例化 `ObjectosphereLoss()` 时保留默认 `sigma=1.0`，而推理 score 又除以 5。训练和推理的边界尺度不一致。
2. **void/unlabeled 的处理与论文文字冲突。** 论文明确说不使用 void/unlabeled 去学习 anomaly；代码却把 `sem_gt == 255` 当作 unknown 拉向特征中心，并把所有 `sem_gt != 255`——包括配置中被 ignore 的 label 0——当作 known 往外推。正常映射下 void 常为 0，因此它仍参与 objectosphere known loss。
3. **合成 OoD 数据的生成代码没公开。** 作者提供了生成好的数据，但论文中的几何插入、beam resampling 和 Lambertian intensity pipeline 标为“available soon”。
4. **推理常数写死。** 例如 norm score 除以 5、每 scan 最大值归一化不走配置，并且缺少全零 score 的保护。

### 判断

**主体算法映射清楚，仓库是所有真实 OOD 模型里最好读、最好改的一个，但有两个会改变论文目标的实现细节。** 把它移植到 occupancy 时，应在 occupied voxel features 上维护 prototype，明确 mask 掉 unobserved、free 和 void，并统一训练/推理半径。

复现评分：**4/5（使用作者生成数据/权重）**；从头生成论文数据则降为 3/5。

---

## 3.3 OccOoD：首个直接 benchmark 很有价值，代码可用性一般

本地材料：`papers/OccOoD.pdf`，`repos/OccOoD/`

### 论文做什么

OccOoD 把 OOD detection 引入 semantic occupancy，贡献包括：

- VAA-KITTI、VAA-KITTI-360 和 VAA-STU；
- cross-space semantic refinement（voxel + BEV geometry/semantic streams）；
- anomaly semantic score（ASS）：对 instance 类使用 feature cosine consistency，对 region 类使用 entropy。

论文在 VAA-KITTI 1.2 m 内报告 AuPRC-r 31.83、AUROC 65.50；SemanticKITTI 和 KITTI-360 的 mIoU 为 16.80 和 18.37。

### 代码核对

CSSR 的 voxel/BEV 双分支、RWKV 融合、logit fusion，以及 ASS 的 instance/region 分治都存在。训练、测试、权重和作者生成的 VAA 数据链接也存在。

但有以下问题：

1. **论文的 VAA 数据生成方法没有源码。** `preprocess/` 只有 depth/image 相关脚本；能下载结果数据，不能复现 realistic anomaly augmentation 本身。
2. **自定义 CUDA extension 使用作者机器绝对路径。** `vrwkv.py` 在 import 时从 `/root/autodl-tmp/OccOoD/...` JIT load `.cpp/.cu`，换机器会直接失败。
3. **KITTI-360 的 ASS 类别分组没有真正切换。** 活跃代码使用 SemanticKITTI 的 instance IDs 1–8、region IDs 9–19；KITTI-360 对应逻辑只是注释块。于是 KITTI-360 的 road/parking 等 ID 会被错误按 instance score 处理。
4. **默认开启保存和 OOD side effects。** head 默认 `save_flag=True`、`ood_flag=True`、`save_flag_ood=True`，validation 会写作者指定目录。
5. **还有绝对数据路径、batch-size-1 假设、无保护 min-max division，以及 `elif step_type == "val" or "test"` 的 Python 逻辑错误。**

### 判断

**论文方法的主干在代码里，但当前仓库更适合拿来做 benchmark/模块参考，不适合原样作为你的 production base。** 特别是 KITTI-360 类别分组问题会改变论文声称的 ASS 行为。

复现评分：**2.5/5**。

---

## 3.4 LiPSOW / OWL：最值得借鉴 object-level unknown grouping

本地材料：`papers/LiPSOW.pdf`，`repos/LiPSOW/`

### 论文做什么

OWL 是 open-world LiDAR panoptic segmentation：

1. 4D-PLS/KPConv 输出 K 个已知类和一个 `other`；
2. 对 thing + other points 用多个 DBSCAN 半径构造层次分割树；
3. PointNet++ 给每个候选 segment 估计 class-agnostic objectness；
4. 在树上选择 objectness 最优的 cut，得到 known/unknown instances。

这比逐 voxel unknown mask 更进一步：它会输出“这里是一个完整未知物体”。论文报告 oracle tree 对 thing objects 的 recall 可到 97.2%，但也明确承认两阶段系统不是实时方法。

### 代码核对

4D-PLS、层次 DBSCAN、PointNet2 objectness 和 recursive tree cut 都存在，树的 parent/children 选择逻辑与论文相符。

工程问题比较严重：

- 顶层 README 基本只说明第一阶段；第二阶段数据生成、PointNet2 训练和端到端调用没有完整命令链；
- 大量 `/project_data/ramanan/...` 绝对路径和固定 DBSCAN eps；
- 推理依赖事先生成的 semantic、instance、embedding、objectness `.npy` 和显式 checkpoint；
- 混合旧 TensorFlow/PyTorch/C++ point ops；静态语法检查发现 6 个 Python 文件无法被 Python 3 parse，其中一个是顶层 `evaluate_conf_matrix.py` 的缩进错误，另外 5 个是遗留 Python 2 test/utility。

### 判断

**论文和核心代码较一致，算法思想值得移植，但复现工程成本最高之一。** 对你的 voxel 输出，优先做 3D connected components + 小型 objectness head；没有必要先完整移植它的旧两阶段栈。

复现评分：**2/5**。

---

## 3.5 ULOPS：论文有潜力，当前仓库不包含论文方法

本地材料：`papers/ULOPS.pdf`，`repos/ULOPS/`

### 论文做什么

ULOPS 使用 Dirichlet evidential semantic decoder，并增加：

- Uniform Evidence Loss；
- Adaptive Uncertainty Separation Loss；
- Contrastive Uncertainty Loss；
- instance embedding、center decoder 和 unknown clustering。

训练使用一小组有标注 proxy unknown，并用严格不重合的测试 vocabulary 评估真正未见类。这个设计比单独 K+1 CE 更合理，因为 unknown 标签用来塑造“低证据/高不确定性”，而不是要求所有 unknown 形成一个单峰类别。

### 仓库事实

仓库有 646 个 Python 文件，看起来很大，但几乎全部是 MMDetection3D 1.0rc4 的底座。全仓检索没有找到论文的 Dirichlet decoder、uniform evidence、adaptive separation 或 contrastive uncertainty loss；也没有 `configs/` 和 README 所链接的 `MODEL_ZOO.md`。README 自己标注为 **Work in Progress** 和 placeholders。

### 判断

**论文上限较高，公开代码与“已经提供 ULOPS reimplementation”的表述不相称。当前无法从该仓库复现论文。** 可以借鉴损失设计，不应把它列为当前可运行 baseline。

复现评分：**0.5/5**。

---

## 3.6 DOODLE：生成式上限有吸引力，但没有公开实现

本地材料：`papers/DOODLE.pdf`，`repos/DOODLE/`

### 论文做什么

DOODLE 在已有 open-set LiDAR model（REAL/LiON）的 range-view semantic features 上训练 VQ-VAE + conditional latent diffusion：

- 只重建 ID feature manifold；
- 以输入和重建 feature 的差异产生 OOD map；
- 放大 baseline redundancy classifier 的 OOD score；
- 通过 HDBSCAN 和 depth-aware density post-processing 去除散点误报。

论文并非完全 anomaly-free：训练时用 OOD mask 把 OOD region 从 diffusion 输入中清掉。LiON + DOODLE 在 SemanticKITTI 上从 AUPR/AUROC 44.68/92.69 提升到 46.53/93.98；nuScenes 从 31.58/95.24 提升到 33.95/95.26。增益存在，但相对于额外 VQ-VAE、diffusion ensemble 和 HDBSCAN，幅度并不大；其表中 post-processing 随保留点比例约需 0.20–1.34 s，40% 设置约 0.77 s。

### 仓库事实

官方仓库只有一个一行 README，0 个 Python 文件，没有模型、训练、评测、配置或权重。

### 判断

**论文提出的方法无法用当前公开仓库核对或复现。** 对在线 occupancy 来说它还带来明显延迟，并依赖一个已有 open-set base，因此当前优先级低。

复现评分：**0/5**。

---

## 3.7 POP-3D：开放词汇 occupancy 实现较一致，但不是 OOD gate

本地材料：`papers/POP3D.pdf`，`repos/POP3D/`

### 论文做什么

POP-3D 使用 camera-only TPVFormer encoder 和两个 head：

- binary occupancy head；
- 每 occupied voxel 输出 CLIP-aligned embedding 的 3D-language head。

训练时用 LiDAR 提供 occupancy supervision，并把 MaskCLIP 的 2D features 沿已注册 LiDAR 点蒸馏到 3D。推理时可以用任意 text prompts 做 zero-shot voxel semantics 或检索。论文报告 class-agnostic occupancy IoU 37.7、zero-shot semantic occupancy mIoU 16.7，以及 retrieval mAP 18.4；训练使用 8×A100、12 epochs。

### 代码核对

TPV aggregator 中确有分别用于 occupancy 和 512-D language feature 的 decoder/classifier；训练脚本联合 CE/Lovász occupancy loss 与 feature MSE；评测代码执行 normalized voxel/text similarity。整体与论文架构一致，并提供论文权重。

主要风险：

1. README 明确标注 SLURM training script **NOT WORKING YET**，虽然直接 `train.py` 入口存在；
2. 依赖 PyTorch 1.10.1/CUDA 11.3/mmcv-full 1.x，feature preprocessing 慢，完整训练资源高；
3. 代码强制 `feature_learning` 和 CLIP features，通用性有限；
4. **训练 occupancy GT 将整个网格初始化为 empty，再把有 LiDAR return 的 voxel 置为 occupied。** 也就是训练时未被 ray 观测的区域也被当成 free。另一个 ray-casting 函数只在 validation label 中把未观测区域设为 ignore。这与论文文字一致，但不适合你已有 LiDAR mapping GT 的设置；你应该让 unobserved 在训练 loss 中保持 ignore。

### 为什么不能直接拿它做 unknown detection

给一组文本后，CLIP similarity 必然产生 argmax；即使输入是从未见过的奇怪障碍物，它也会被分给最相似的 prompt。它没有受校准的“none of the above”概率，也没有保证 OOD score 与 open-vocabulary similarity 等价。

### 判断

**适合在 unknown gate 后做命名，不适合单独负责拒识。** 论文—代码一致性较好，但复现资源和旧依赖成本较高。

复现评分：**2.5/5**。

---

## 3.8 EvOcc：你的 GT 构建最应该借鉴的论文

本地材料：`papers/EvOcc.pdf`，`repos/EvOcc/`

EvOcc 不是 object OOD detector，而是 evidential semantic occupancy mapping：

- LiDAR ray 的 **transmission** 给 free evidence；
- 有 semantic label 的 return 给某一 class occupied evidence；
- 无 semantic label 的 return 给 generally occupied evidence；
- 没有证据或证据冲突用 belief mass / $\Omega$ 表达，而不是强行 free 或某一类；
- 多帧时对静态环境和动态物体做 motion compensation 后累积证据。

这恰好把你的 `generic occupied` 与 `unobserved` 分开：

- unlabeled LiDAR hit：`occupied, semantics unknown`；
- ray 从未经过：`unobserved/ignorance`；
- ray 经过且在 return 前：`free`。

论文报告 ray-based mIoU 相比 baseline 的 GT 至少提升 15.8%，训练模型至少提升 5.5%。但官方仓库只有 README、图片和 license，并明确写着 code will be available soon。

判断：**方法思想应直接进入你的 GT pipeline，但需要自己实现。** 复现评分：**0.5/5**。

---

## 3.9 STU：最好的真实 LiDAR unknown 测试资产

本地材料：`papers/STU.pdf`，`repos/STU/`

STU 是 128-beam real-world LiDAR anomaly segmentation dataset，而不是 occupancy dataset。仓库包含：

- 数据下载和 SHA256；
- point-level AP、FPR95、AUROC evaluator；
- object-level UQ/PQ evaluator；
- DBSCAN instance 生成；
- Mask4Former3D baseline 训练、推理和 checkpoints。

point evaluator 只评估 2.5–50 m，并跳过 anomaly 点少于 5 个的 frame；这意味着指标是论文协议下、对 anomaly-bearing frames 的条件性能，与你自己的全序列 false alarm rate 不是同一个量。

复现障碍主要是 MinkowskiEngine：README 说明 CUDA 12 需要额外 patch，CUDA 12.8 构建失败；训练只支持单卡且作者使用 H100。validation 可自行评测，test 目前需把约 3 GB prediction 链接交给作者运行。

判断：**非常适合作为真实 unknown 泛化测试，但需把 point mask voxelize，并另建 occupancy/free-space GT。**

---

## 3.10 CODA：有 generic/corner objects，但不是 3D occupancy 数据

本地材料：`papers/CODA.pdf`，`repos/CODA/`

官方 CODA2022 含 9,768 张图、80,180 个 2D bounding boxes、43 类；前 7 类为 common，其余为 novel。validation 只覆盖 29 类，test 覆盖全部 43 类，用来模拟部署后出现全新类别。官方站点仓库包含标注文档、下载页和 COCO-style evaluator。

它的局限是：核心监督是 camera 2D box，不提供直接的 3D voxel semantics/free-space labels。它适合：

- 设计 generic-object taxonomy；
- camera-side corner-case detector；
- 验证开放词汇命名；

但不能直接训练或评测你的 LiDAR occupancy unknown head。

---

## 4. 给你的最合理 pipeline

## 4.1 Ground truth：先把可见性做对

建议每个 voxel 保存两个 GT 字段，而不是一个互斥 label：

```text
geometry_state ∈ {FREE, OCCUPIED, UNOBSERVED}
semantic_state ∈ {known class 1..K, GENERIC, NONE}
```

构建规则：

1. 对每条 LiDAR ray，从 sensor origin 到 first return 之前的体素累积 free evidence；
2. first return 所在 voxel 和适当的 surface thickness 邻域累积 occupied evidence；
3. 有可靠类别 label 的 return 写入 known semantic evidence；
4. 无类别 label 的真实背景 return，以及你插入的 sofa/tire/fallen trunk，写入 `occupied + GENERIC`；
5. 没有 transmission 或 reflection evidence 的 voxel 保持 `UNOBSERVED`，不参与 free/occupied CE；
6. 多帧 mapping 先做 ego pose 对齐；动态车/人要用 object motion compensation，或限制时间窗口，防止拖影变成 occupied GT；
7. 有 free/occupied 冲突时保留 evidence/conflict 或降低 loss weight，不要简单以后到者覆盖。

如果不想完整实现 Dempster-Shafer，至少保存：

```text
free_count, occupied_count, semantic_count[K], generic_hit_count, observed_mask
```

然后将 count 转成 soft target 和 confidence weight。这样已经比二值 overwrite 稳健很多。

## 4.2 网络：两个主 head + 一个 novelty score

可以保持你喜欢的两个主 head：

1. **Geometry head**：预测 $P(occupied)$；只在 `observed_mask=1` 的 voxel 训练。
2. **Known semantic head**：输出 K 个已知类 logits；只在 `occupied + known semantic` voxel 上训练。

再从 semantic/neck feature 计算一个 **novelty branch/score**。它可以只是轻量 MLP、prototype distance 或 evidential scalar，不必变成完整第三个 decoder：

$$
s_{ood} = f\left(1-\max_k \cos(z,p_k),\ H(p(y|z)),\ \|z\|,\ s_{local}\right).
$$

部署规则：

```text
if p_occ < tau_occ:
    FREE
elif s_ood > tau_ood:
    GENERIC_OBJECT
else:
    argmax known semantic class
```

`UNOBSERVED` 是输入/预测置信度状态或单独 mask，不能由 `FREE` 代替。

## 4.3 如何使用 generic object 标注

训练中可以使用 generic labels，但用途应是 **边界塑形**，而不是普通 K+1 CE：

- generic voxel 继续监督 geometry head 为 occupied；
- generic voxel不进入 K 类 semantic CE；
- 用它监督高 novelty / 低 known evidence；
- 对 generic features 使用 entropy/evidence margin 或 energy loss，不要求不同 generic objects 相互接近；
- sofa、tire、tree trunk 最好保留 instance ID，供 object-level grouping/objectness loss 使用。

最重要的一项训练策略是 **episodic class holdout**：每轮把一部分原有 known classes 暂时当 pseudo-unknown，它们仍监督 occupied，但不参与当轮 known semantic CE，并监督 novelty head。这样模型学到的是“相对于当前已知集合之外”的规则，而不是只记住你插入的几个 asset domain。

## 4.4 推荐损失

$$
\mathcal L =
\lambda_{occ}\mathcal L_{occ}
+\lambda_{sem}\mathcal L_{known-sem}
+\lambda_{proto}\mathcal L_{proto}
+\lambda_{nov}\mathcal L_{known/unknown-margin}
+\lambda_{inst}\mathcal L_{objectness}.
$$

- $\mathcal L_{occ}$：observed free/occupied 上的 focal/BCE + Lovász；
- $\mathcal L_{known-sem}$：仅 known occupied voxel 的 CE/Lovász；
- $\mathcal L_{proto}$：LIDO/ProOOD 式 known class compactness，EMA prototype 只从高置信 true positives 更新；
- $\mathcal L_{known/unknown-margin}$：让 held-out/generic 的 known-support score 显著低于 known；
- $\mathcal L_{objectness}$：可选，对 class-agnostic connected components 学完整物体质量。

不要对 unobserved voxel 施加任何 hard free loss，也不要把自然背景中“没有 semantic annotation”的 occupied return 当 free。

## 4.5 推理与 open vocabulary

建议分两级：

1. **安全层**：geometry + calibrated novelty，回答“这里有障碍物，且不属于可靠已知类”。这层不依赖文本 prompt。
2. **解释层**：只对已判为 generic 的 component 聚合 3D feature，再用 CLIP/开放词汇模型查询 `sofa`、`tire`、`fallen tree trunk` 等名称。输出名称和 similarity，但保留 `unknown` 安全标签。

对象级 aggregation 很重要。单 voxel anomaly score 容易受边界、距离和稀疏性影响；在 3D component 内对 score 做 trimmed mean/top percentile，并增加最小体积、地面接触、temporal persistence，会显著减少散点误报。

## 4.6 Calibration 和评测

threshold 不应在测试 unknown 类上调。用独立 validation classes 做 leave-category-out calibration，并固定 known false-positive rate，例如：

- known voxel FPR = 1% 或 5%；
- class-conditional threshold，避免 rare known classes 被大量报 unknown；
- 必要时对距离区间分别 calibration。

至少同时报告：

- observed-region occupancy IoU；
- known semantic mIoU；
- unknown voxel AuPRC、AUROC、FPR95；
- OccOoD 的 distance-aware AuPRC-r；
- unknown object recall / UQ / component AP；
- 每公里 false-positive objects；
- 按距离、物体大小、类别拆分；
- synthetic inserted objects 与真实 STU anomalies 分开报告。

不要只报告 AUROC：unknown voxel 极少时，AUROC 看起来不错但 precision 可能很差，AuPRC 和 object-level false alarms 更接近实际部署需求。

---

## 5. 建议的实现顺序

1. **先修 GT**：ray transmission/free、return/occupied、unobserved ignore、generic occupied 分开；这是收益最大且不依赖模型的步骤。
2. **建立最小 baseline**：现有 occupancy backbone + geometry head + known semantic head，增加 MSP、energy、prototype cosine distance 三个无需复杂训练的 OOD baseline。
3. **移植 LIDO prototype/objectosphere 思路**：修正 void mask 和半径，作用在 occupied voxel feature 上。
4. **加入 ProOOD 的 EMA prototype 和 3D local consistency**：自己实现 Eq. 8，而不是直接复制当前 unreachable top-2 分支；避免 per-scene min-max 后简单 max，改用 validation-calibrated fusion。
5. **episodic held-out + generic insertion 训练 novelty**：generic 只监督 occupied 与 novelty，不做普通 K+1 聚类。
6. **object-level grouping**：先用 connected components，必要时再加入 LiPSOW 式 objectness/tree。
7. **最后接 open vocabulary**：对已拒识的 generic components 命名；不让 CLIP 决定安全层的 unknown 与否。

## 最终推荐

以论文为标签来概括，最值得做的组合是：

```text
EvOcc-style visibility/evidence GT
        +
your two-head occupancy network
        +
LIDO known-support / prototype scoring
        +
corrected ProOOD voxel prototype + local coherence
        +
LiPSOW-inspired object grouping
        +
POP-3D/CLIP naming after rejection
```

这条路线比“semantic head 增加一个 unknown 类”多一点实现量，但它把几何占据、未观测、已知语义、未知拒识和未知命名正确拆开了；同时每个模块都有独立 metric，可以知道失败究竟来自 occupancy、known classification 还是 OOD calibration。

---

## 6. 审计边界

本次已经完成：

- 10 篇官方论文逐篇阅读；
- 10 个官方代码/资产仓库下载并固定 commit；
- 方法模块、loss、inference score、训练/评测入口和 README 声明的源码级交叉检查；
- 全部 Python 文件静态语法解析：ProOOD、LIDO、OccOoD、POP3D、STU、ULOPS 无语法错误；LiPSOW 有 6 个遗留/顶层文件错误；DOODLE/EvOcc 无 Python 实现。

尚未做：下载各数据集和 checkpoints、构建十套互相冲突的 CUDA 环境、完整训练或逐项复跑论文数字。因此这里的“一致/不一致”是**代码结构与论文算法声明的一致性审计**，不是对作者全部实验数字的独立复现认证。

完整下载地址、commit 和文件统计见 `MANIFEST.md`。
