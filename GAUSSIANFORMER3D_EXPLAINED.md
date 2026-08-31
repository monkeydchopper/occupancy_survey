# GaussianFormer3D：论文、代码与 OOD Occupancy 视角下的完整解析

> 论文：**GaussianFormer3D: Multi-Modal Gaussian-based Semantic Occupancy Prediction with 3D Deformable Attention**  
> 作者：Lingjun Zhao, Sizhe Wei, James Hays, Lu Gan  
> 状态：ICRA 2026；最初发表于 RSS 2025 Gaussian Representations for Robot Autonomy Workshop  
> 本文核对范围：论文 v2、官方项目页、公开代码与配置；最后核对日期为 2026-08-29。

---

## 0. 先给结论

GaussianFormer3D 是一套**真正把 LiDAR 和多目相机同时用于推理**的 Gaussian semantic occupancy 模型。它最重要的设计不是“用 LiDAR 点直接做语义”，而是让 LiDAR 承担两件几何工作：

1. 用非空 LiDAR voxel 的平均坐标和强度，初始化 Gaussian 的位置和 opacity；
2. 把 LiDAR 投影深度与相机特征组成带深度约束的 3D feature，再让每个 Gaussian query 从中采样视觉信息。

一句话概括它的计算流：

> **LiDAR 先告诉 Gaussian 应该在哪里，并告诉相机 feature 应该落在哪个深度；Camera 再提供主要的可学习外观/语义 feature；Gaussian 经四轮稀疏交互和 refinement 后，被 splat 回 voxel occupancy。**

它和我们前面研究的 S2GO 有本质区别：

- GaussianFormer3D 是 **one query ↔ one Gaussian**；
- S2GO 是 **one parent query → multiple child Gaussians**；
- GaussianFormer3D 没有 learned temporal memory、Past/New/Future query，也没有 entity 层；
- GaussianFormer3D 的 LiDAR 在正常 inference 中一直存在，S2GO 原版则是 camera-only inference。

对我们的 OOD occupancy 课题，我的判断是：

> **GaussianFormer3D 比 S2GO 更适合作为第一版可复现的 LiDAR–camera Gaussian backbone；S2GO 更适合提供 parent/entity hierarchy 和 temporal memory 的研究结构。最合理的路线是以 GaussianFormer3D 的 V2G + 3D deformable fusion 为底座，再引入视觉主导的 open-semantic feature、entity 层和时序记忆。**

但不要高估它目前的能力：它是 fully supervised、closed-set、single-frame model；论文主模型在一张 A40 上报告 **555 ms/frame，约 1.8 FPS**，远非实时。

---

## 1. 它要解决什么问题？

### 1.1 输入和输出

在 nuScenes 配置中，模型输入是：

- 6 路环视 RGB 图像；
- 当前 LiDAR 点云和最多 10 个历史 sweeps；
- 相机内外参及 LiDAR–camera 标定。

输出是一个语义 occupancy grid：

$$
\hat O \in \mathbb{R}^{200\times200\times16\times18},
$$

覆盖范围为：

$$
[-50,50]\text{ m}\times[-50,50]\text{ m}\times[-5,3]\text{ m},
$$

voxel 分辨率为 $0.5$ m。18 个输出类别包括 17 个非空语义通道和 1 个 empty 通道。

### 1.2 论文真正想解决的两个痛点

Camera-only Gaussian occupancy 已经有 GaussianFormer、GaussianFormer-2 等工作，但会遇到两个问题：

1. **Gaussian 初始化没有可靠几何先验。** 随机或 learnable anchors 必须只靠图像把自己移动到正确的 3D 表面，优化困难。
2. **2D deformable attention 有深度歧义。** 不同深度的多个 Gaussian 可能投影到同一个 image pixel，从而读取几乎相同的 2D feature。

GaussianFormer3D 的两个对应模块就是：

- **V2G：Voxel-to-Gaussian initialization**；
- **DFA：LiDAR-guided 3D Deformable Attention**。

论文消融非常清楚：

| V2G | DFA | IoU | mIoU |
|---:|---:|---:|---:|
|  |  | 29.2 | 18.8 |
| ✓ |  | 40.7 | 25.8 |
|  | ✓ | 40.7 | 26.4 |
| ✓ | ✓ | **43.3** | **27.1** |

二者单独加入都带来巨大提升，组合后继续互补。这说明 LiDAR 不只是改善了一点 depth，而是同时改变了 **query 初始位置**和 **feature retrieval** 两个环节。

---

## 2. 总体模型图

```text
                     GaussianFormer3D

  Multi-view RGB                                  Multi-sweep LiDAR
       │                                                   │
       ▼                                                   ├──────────────┐
  R101-DCN + FPN                                           │              │
       │                                                   ▼              ▼
  4-scale image features                            fine voxelization   project to cameras
       │                                            0.075×0.075×0.2 m      │
       │                                                   │         sparse depth maps
       │                                                   ▼              │
       │                                      Voxel-to-Gaussian (V2G)     │
       │                                      mean xyz → Gaussian center  │
       │                                      mean intensity → opacity    │
       │                                                   │              │
       │                                                   ▼              │
       │                                          25,600 Gaussians        │
       │                                          + latent queries Q      │
       │                                                   │              │
       └───────────────────────────┬───────────────────────┘              │
                                   │                                      │
                                   ▼                                      │
                         4 × iterative update blocks                      │
                 ┌──────────────────────────────────────────┐             │
                 │  Gaussian keypoints                      │             │
                 │       ↓ project to (u,v,d)               │             │
                 │  LiDAR-guided 3D deformable attention ◄──┼─────────────┘
                 │       ↓                                  │
                 │  FFN / sparse 3D convolution             │
                 │       ↓                                  │
                 │  refine center, scale, rotation,         │
                 │  opacity and semantic coefficients       │
                 └──────────────────────────────────────────┘
                                   │
                                   ▼
                        final semantic Gaussians
                                   │
                                   ▼
                    Gaussian-to-Voxel local splatting
                                   │
                                   ▼
                      200 × 200 × 16 occupancy logits
```

这张图里最值得注意的是：模型没有构造一个完整的 dense 3D BEV feature volume，再在上面跑 3D CNN。它保留 image feature 和 sparse depth distribution，通过定制的 3D deformable-attention CUDA operator，只在 Gaussian 需要的位置进行采样。

---

## 3. Gaussian 与 query 到底是什么关系？

### 3.1 每个 Gaussian 的显式物理属性

第 $i$ 个 Gaussian 具有：

$$
G_i=(m_i,s_i,r_i,\sigma_i,c_i),
$$

其中：

- $m_i\in\mathbb{R}^3$：3D center；
- $s_i\in\mathbb{R}^3$：三个主轴尺度；
- $r_i\in\mathbb{R}^4$：四元数旋转；
- $\sigma_i\in[0,1]$：opacity / occupied contribution；
- $c_i\in\mathbb{R}^{17}$：前景语义系数。

对应的空间贡献可以写成：

$$
g(x;G_i)=\sigma_i
\exp\left[-\frac{1}{2}(x-m_i)^\top\Sigma_i^{-1}(x-m_i)\right]c_i,
$$

其中 $\Sigma_i$ 由 scale 和 rotation 组成。于是某个位置 $x$ 的语义 logit 是附近 Gaussians 的贡献之和：

$$
\hat O(x)\approx\sum_{i\in\mathcal N(x)}g(x;G_i).
$$

### 3.2 每个 Gaussian 还有一个 latent query feature

模型同时维护：

$$
Q=\{q_i\in\mathbb{R}^{128}\}_{i=1}^{N_g}.
$$

$q_i$ 不直接表示 center、scale 或语义类别。它是 attention 和 refinement 使用的隐变量，负责从图像中聚合信息，再由 MLP 解码出 $G_i$ 的显式属性。

因此有两个平行状态：

```text
latent state q_i             physical state G_i
“这个 primitive 知道什么”    “这个 primitive 在哪里、长什么样、是什么”
          │                              ▲
          ├── attention / sparse conv ───┤
          └──────── refinement MLP ──────┘
```

这里是严格的 **one-to-one**：

$$
q_i\longleftrightarrow G_i.
$$

它不是 S2GO 中“一条 query 生成一簇 Gaussians”的父子结构。

### 3.3 这个 query 是 object / instance / entity 吗？

不是。它更接近一个 **Gaussian primitive token**：

- 没有 instance GT matching；
- 没有一条 query 对应一个完整物体的约束；
- 没有 objectness 或 bbox head；
- 没有 query ID 的跨帧延续；
- 一个物体通常由许多 Gaussian queries 共同覆盖；
- 同一 Gaussian 也可能落在物体边界附近。

所以 GaussianFormer3D 依然是 **semantic occupancy segmentation**，不是 3D object detection，也不是 panoptic occupancy。

---

## 4. Voxel-to-Gaussian：LiDAR 如何产生 Gaussian 初始位置？

### 4.1 多帧点云先做 ego 对齐和细粒度 voxelization

nuScenes 主配置会加载当前点云与最多 10 个历史 sweeps，并转换到当前 LiDAR frame。随后使用很细的 voxel：

$$
0.075\text{ m}\times0.075\text{ m}\times0.2\text{ m}.
$$

这不是最终 occupancy 的 $0.5$ m voxel。它只是用来把密集、重复的多 sweep 点压缩成候选 Gaussian seeds。

对每个非空 voxel $v$，HardSimpleVFE 计算其中点特征的均值。论文使用平均坐标和平均 intensity：

$$
m_i=\frac{1}{|P_v|}\sum_{j\in P_v}(x_j,y_j,z_j),
$$

$$
\sigma_i=\frac{1}{|P_v|}\sum_{j\in P_v}\eta_j.
$$

代码进一步把坐标映射到 $[0,1]$ 的 occupancy range，再做 inverse-sigmoid，写入内部 anchor 参数；intensity 会先除以 255，再经 inverse-sigmoid 写入 opacity slot。

### 4.2 非空 voxels 与固定 Gaussian budget 如何匹配？

主模型固定：

$$
N_g=25{,}600.
$$

设当前帧非空 LiDAR voxels 数量为 $N_v$：

- 若 $N_v\geq N_g$：随机抽取 $N_g$ 个非空 voxels，初始化全部 Gaussians；
- 若 $N_v<N_g$：随机挑选 $N_v$ 个 Gaussian slots 用 LiDAR 初始化，其余保留模型原来的随机/learnable anchors。

这一步很像“生成 spatial proposals”，但必须准确理解：

> **V2G 提出的是 occupied-surface primitive candidates，不是 object proposals。**

一个 LiDAR voxel 对应一个初始 Gaussian；它不试图圈出一个完整物体，也没有 NMS、bbox 或实例 matching。

### 4.3 为什么 voxel seed 比 raw LiDAR point seed 好？

论文消融：

| 初始化方式 | IoU | mIoU |
|---|---:|---:|
| Single-sweep point | 36.7 | 22.4 |
| GaussianFormer-2 的 PM-Point | 34.9 | 21.2 |
| **Multi-sweep voxel** | **40.7** | **25.8** |

原因包括：

- 多 sweep 增加远处、小物体和稀疏区域的几何覆盖；
- voxel averaging 降低点级噪声和重复点；
- 固定 query budget 下，voxel 比 raw point 提供更均匀的空间候选；
- Gaussian center 一开始就在 occupied surface 附近，避免从空空间长距离搜索。

### 4.4 intensity 作为 opacity 合理吗？

这是一个有效但比较粗糙的先验。intensity 不是“这里被占据的概率”：它同时受材质、入射角、距离、传感器型号和天气影响。将其用于 opacity initialization 可以提供反射强弱线索，但不应该让网络把 intensity 当成 objectness 或 semantic evidence。

对 OOD 尤其要小心：罕见材质、黑色物体、反光标志、雨雾点等都可能产生异常强度分布。更稳妥的扩展是：

$$
\text{LiDAR seed feature}
=\operatorname{MLP}(xyz,\eta,\text{return count},\text{sweep age},\text{local geometry}),
$$

并让 opacity / existence 由网络预测，而不是直接把强度当作物理占据置信度。

### 4.5 一个值得实际检查的代码边界问题

公开代码在 $N_g>N_v$ 的分支使用了类似：

```python
anchor[batch_idx][idx][:, :3] = voxel_coords
```

在 PyTorch 中，连续的 advanced indexing 可能先生成临时 tensor，再对临时 tensor 赋值，而没有写回原始 `anchor`。常规 nuScenes 多 sweep 配置下，细粒度非空 voxels 很可能多于 25,600，因此主实验通常走另一个分支；但若用于稀疏 LiDAR、裁剪区域或低线数雷达，这个分支应该单独写测试确认，并改成一次性索引赋值。

这不是论文方法问题，但会影响我们迁移到稀疏传感器时的可靠性。

---

## 5. LiDAR-guided lifted 3D feature：它到底怎样融合 Camera 和 LiDAR？

### 5.1 Camera branch

nuScenes 主模型使用：

- ResNet-101 + DCNv2；
- FCOS3D 预训练 checkpoint；
- FPN 输出 4 个尺度；
- 输入图像约为 $900\times1600$；
- feature dimension 为 128。

对第 $c$ 个 camera、第 $l$ 个尺度，可记图像 feature 为：

$$
F^{c,l}_{\text{img}}(u,v)\in\mathbb{R}^{128}.
$$

### 5.2 LiDAR depth branch不是 monocular DepthNet

这是阅读摘要时最容易误判的地方。

主配置使用 `DepthHead_GTDpt`。它没有根据 RGB 学习 dense monocular depth，而是读取**提前由 LiDAR 投影到各相机平面的 sparse depth map**。训练和测试 pipeline 都会加载这些 depth 文件。

深度范围和 bin 是：

$$
d\in[2,58)\text{ m},\qquad\Delta d=0.5\text{ m},
$$

因此共有：

$$
N_d=\frac{58-2}{0.5}=112
$$

个 depth bins。

每个下采样 image cell 中选择最近的有效 LiDAR depth，转成 one-hot depth distribution。配置 `max_tol=2` 会把有效 bin 向两侧扩展两个 bins，相当于给稀疏深度约约 $\pm1$ m 的离散容差。

因此：

$$
F^{c,l}_{d}(u,v,d)\in\{0,1\}.
$$

没有 LiDAR return 的 image region 基本没有有效 depth support。

### 5.3 outer product 是什么意思？

论文写作：

$$
F^{3D}=F_d\otimes F_{\text{img}}.
$$

对一个 pixel，可理解为：

$$
F^{3D}(u,v,d,:)
=F_d(u,v,d)\cdot F_{\text{img}}(u,v,:).
$$

直觉图：

```text
同一个 image pixel (u,v)

camera feature:      [car-like visual feature]

沿 ray 的 depth bins:
  10 m    15 m    20 m    25 m    30 m
   0       0       1       0       0      ← LiDAR depth support
                   ×
          [car-like visual feature]
                   ↓
只有 20 m 附近的 3D sampling point 能读到这组视觉 feature
```

这样，两个投影到相同 $(u,v)$、但深度分别是 20 m 和 35 m 的 Gaussians，不再读取同样的 feature。35 m 处如果没有 depth support，采样结果会很弱或为零。

### 5.4 它是否真的构造巨大的 dense 3D tensor？

概念上可以把它写成 $F^{3D}(u,v,d)$，但代码没有必要显式存储所有 $H\times W\times112\times128$ feature。自定义 DFA3D operator 分别接收：

- flattened multi-scale image feature；
- flattened depth distribution；
- $(u,v,d)$ sampling locations；
- attention weights。

它在采样时才把视觉 value 与相应 depth score 结合。也就是说，这是一个**稀疏查询式的 3D fusion**，不是完整 dense volume construction。

### 5.5 这算 camera 主导吗？

它可以被称作“**visual value + LiDAR geometric gate**”：

- 被聚合的高维 value 主要来自 camera feature；
- LiDAR 决定这个 visual value 在哪个 depth 上有效；
- LiDAR 还决定大部分 Gaussian 初始位置；
- 没有强 LiDAR semantic backbone 给出类别 feature。

因此它符合我们想要的“LiDAR 提供距离、Camera 提供主要视觉语义”的大方向。

但它不是完全的 camera-dominant design，因为 LiDAR depth 在 feature lifting 中是乘法门控。没有 LiDAR return 的区域，即使图像里存在清晰物体，visual feature 也很难进入相应 3D query。这一点对远距离、小物体、低反射 OOD 和 LiDAR 遮挡区域尤其危险。

---

## 6. 3D Deformable Attention：每个 Gaussian 怎样读取融合特征？

Gaussian center 只是一个点，而一个椭球的空间范围可能很大。只在 center 取 feature 容易错过边界和局部结构。因此模型采用两阶段 sampling。

### 6.1 第一阶段：在 Gaussian 自身坐标系内取 keypoints

默认有 7 个固定 keypoints：

```text
                    +z
                     ●
                     │
              -y ●── ○ ──● +y
                    /│
                  ●  │
                -x   ● +x / -z 方向同理

○ = Gaussian center
● = 沿三个主轴正负方向的采样点
```

实际固定模板为：

$$
(0,0,0),\quad(\pm0.45,0,0),\quad(0,\pm0.45,0),\quad(0,0,\pm0.45).
$$

这些模板会先乘 Gaussian scale，再乘 rotation，因此采样点跟着椭球的形状和朝向变化。另外还从 query feature 学习 2 个自适应 keypoints，所以每个 Gaussian 一共得到：

$$
N_{R1}=7+2=9
$$

个 3D reference points。

形式上：

$$
r_i=m+R\big(s\odot\Delta m_i\big),\qquad i=1,\ldots,9.
$$

### 6.2 投影到每个 camera 的 $(u,v,d)$ 空间

通过相机外参 $T_c$ 和内参 $K_c$：

$$
(u_i^c,v_i^c,d_i^c)=\pi_c(r_i;T_c,K_c).
$$

这里与普通 2D deformable attention 的关键差别是保留了 $d$，而不是只保留 $(u,v)$。

### 6.3 第二阶段：在投影后的 3D image space 再学 offsets

模型还会从 query feature 预测：

$$
\Delta \bar m_{ij}
=(\Delta u_{ij},\Delta v_{ij},\Delta d_{ij}).
$$

配置中每个 keypoint 有 2 个 sampling offsets，因此每个 Gaussian、每个 feature level 的 sampling positions 数量是：

$$
N_{R1}\times N_{R2}=9\times2=18.
$$

再考虑 4 个 FPN levels、4 个 attention groups 和 6 个 cameras，模型在很多局部候选上做加权聚合，但仍然只围绕 25,600 个 Gaussian queries 采样。

### 6.4 Attention 更新

对一个 query，可将更新抽象为：

$$
\Delta q
=\frac{1}{N_c}
\sum_{c=1}^{N_c}
\sum_{l,i,j,h}
\alpha_{c,l,i,j,h}
\operatorname{Sample}
\left(F^{3D}_{c,l},\bar m_{ij}^{c}\right),
$$

其中 $h$ 是 feature group，$\alpha$ 由 $q$、Gaussian anchor embedding 和 camera embedding 共同产生，并在每个 camera 内的 level × sampling-point 维度归一化。公开主配置关闭 `use_visibility`，各相机输出最终直接求均值；这比显式按可见 camera 数量归一化更简单，也值得我们后续重新检查。

官方主配置继承了 `residual_mode="cat"`，聚合后实际先做：

$$
q_{\text{attn}}=[W_o\Delta q\,\|\,q],
$$

得到 256 维拼接特征，再由 Asymmetric FFN 压回 128 维。也就是说，旧 query 没有丢掉，但实际融合是 concat + FFN，而不是单纯相加。

### 6.5 两阶段 offset 是否真的有用？

| Gaussian space 投影前 offset | $(u,v,d)$ 投影后 offset | IoU | mIoU |
|---:|---:|---:|---:|
| ✓ |  | 37.7 | 24.5 |
|  | ✓ | 40.1 | 26.1 |
| ✓ | ✓ | **40.7** | **26.4** |

第二阶段更加关键，说明校准误差、离散 depth bin 和投影后的局部 feature mismatch 确实存在；两个阶段组合最好。

### 6.6 为什么不直接把 sparse depth 和 RGB channel concat？

论文比较：

| 特征融合方式 | IoU | mIoU |
|---|---:|---:|
| 2D RGB + sparse depth concat | 36.1 | 22.2 |
| 2D RGB + completed dense depth concat | 36.6 | 22.1 |
| **3D deformable attention** | **40.7** | **26.4** |

2D concat 只能告诉网络“这个 pixel 有多深”，但多个 3D queries 仍然需要在 2D 平面竞争同一位置的 feature。$(u,v,d)$ attention 把 query 的 3D hypothesis 和 LiDAR depth support 直接放在同一个 sampling coordinate system 中，处理的是匹配问题，而不只是 feature concatenation。

---

## 7. 四轮 Gaussian update 具体怎样进行？

模型共有 4 个 refinement blocks。

第一轮：

```text
3D deformable attention → FFN → LayerNorm → Gaussian refinement
```

后三轮：

```text
sparse 3D convolution → LayerNorm
→ 3D deformable attention → FFN → LayerNorm
→ Gaussian refinement
```

### 7.1 Sparse 3D convolution 的作用

Deformable attention 让每个 Gaussian 从 camera–LiDAR feature 中读取证据；SparseConv 则让空间邻近的 Gaussians 彼此交换信息。

可以理解为：

```text
cross-modal observation                  local 3D context
image + depth ──► Gaussian query ◄── neighboring Gaussian queries
```

公开配置使用 $0.5$ m sparse grid。它不是对最终 occupancy grid 做 dense convolution，而是把 query 按当前 center 临时散列到 sparse grid 上，进行 submanifold sparse convolution。

### 7.2 Refinement head 更新什么？

Refinement MLP 输入：

$$
q_i+E(G_i),
$$

输出新的 center、scale、rotation、opacity 和 semantic coefficients。

代码对 center 使用 residual update：

$$
\tilde m_i^{t+1}
=\tilde m_i^t+\Delta\tilde m_i^t,
$$

并限制单个 block 的最大物理位移约为：

$$
(4\text{ m},4\text{ m},1\text{ m}).
$$

其余属性不是简单 residual add，而是每轮重新预测：

- scale 被约束到 $[0.01,1.8]$ m；
- rotation 做四元数归一化；
- opacity 经 sigmoid；
- semantic coefficients 经 softplus，而不是 Gaussian-level softmax。

Softplus 允许一个 Gaussian 对多个类别通道产生非负贡献；最终 voxel logits 是许多 Gaussians 累加后再进入 CE，而不是先强迫每个 Gaussian 单独做归一化分类。

---

## 8. Gaussian-to-Voxel：连续表示如何变回 occupancy？

### 8.1 只用 25,600 个 Gaussians 表示 occupied space

主模型的预测 Gaussians 只有 17 个 occupied semantic channels，并不让它们大量铺满 empty space。空空间由额外一个固定大 Gaussian 表示：

$$
m_{\text{empty}}=(0,0,-1),
$$

$$
s_{\text{empty}}=(100,100,8),
$$

rotation 为单位四元数，opacity 为 1，只在 empty channel 上有一个可学习 scalar。

这个技巧很聪明：empty voxel 占整个场景绝大多数，如果也用许多小 Gaussian 表示，会极大浪费 primitive budget。

### 8.2 Local Gaussian aggregation

对每个待评估 voxel center $x$，只聚合邻域中的 Gaussians：

$$
\hat O(x)
=\sum_{i\in\mathcal N(x)}
\sigma_i
\exp\left[-\frac12(x-m_i)^\top\Sigma_i^{-1}(x-m_i)\right]c_i.
$$

CUDA local aggregator 用每个 Gaussian 最大 scale 的 3 倍估算离散影响半径：

$$
r_i=\left\lceil
\frac{3\max(s_i)}{0.5}
\right\rceil.
$$

所以复杂度依赖局部覆盖，而不是对 $640,000$ 个 voxels 与 $25,600$ 个 Gaussians 做完整笛卡尔积。

### 8.3 连续 Gaussian 的额外优点

同一组 Gaussians 可以在不同 voxel resolution 上评估。论文展示了无需重新训练，直接输出：

- $200\times200\times16$，0.5 m；
- $500\times500\times40$，0.2 m。

但要注意：更高分辨率只是对已学 continuous field 更密集地采样，不会凭空恢复传感器未观察到的真实细节。

### 8.4 代码实现的性能与可移植性注意项

G2V 是自定义 CUDA extension，并把类别数编译进 `config.h`。从 nuScenes 的 18 类切到 WildOcc 的 9 类时，需要修改宏并重新编译。

另外，公开 head 中有一行：

```python
CovInv = Cov.cpu().inverse().cuda()
```

这会把每帧所有 Gaussian covariance 从 GPU 搬到 CPU 求逆，再搬回 GPU，存在明显的延迟、设备绑定和 portability 风险。我们若基于此开发，应先改为 device-preserving 的 `torch.linalg.inv` 或解析 3×3 inverse，并做数值稳定性测试。

---

## 9. 训练监督与 inference

### 9.1 监督目标

这是标准的 fully supervised semantic occupancy training。最终 Gaussian field 被 splat 到所有 voxel centers，与 dense occupancy GT 比较：

$$
\mathcal L
=10\mathcal L_{\text{weighted CE}}
+\mathcal L_{\text{Lovasz}}.
$$

公开主配置：

- 训练 24 epochs；
- AdamW，learning rate $10^{-4}$；
- weight decay 0.01；
- image backbone learning-rate multiplier 为 0.1；
- `apply_loss_type='random_1'` 实际只对最后一个 decoder output 施加 occupancy loss；
- Lovasz loss 忽略 empty label。

所以它**需要 occupancy GT**。论文也明确把 fully supervised 标为主要 limitation。

### 9.2 inference 时 LiDAR 做什么？

全部都做：

1. 读取当前点云和历史 sweeps；
2. V2G 初始化 Gaussian centers 与 opacity；
3. 把 LiDAR 投影成 sparse multi-view depth maps；
4. 在 3D deformable attention 中对 camera feature 做 depth gating；
5. 输出 occupancy。

这里不存在“S2GO 那样训练用 LiDAR、推理不用 LiDAR”的切换。

### 9.3 它是不是 temporal model？

不是。

虽然输入聚合了历史 LiDAR sweeps，但网络没有：

- past-query queue；
- temporal attention；
- persistent query identity；
- velocity head；
- future rollout；
- learned temporal feature memory。

历史 sweeps 只是被 ego-motion 对齐后合并成当前时刻的静态点云输入。动态物体的历史点可能留下拖影；公开 loader 没有显示逐物体 motion compensation。因此应把它叫作 **multi-sweep geometric aggregation**，而不是 temporal occupancy modeling。

---

## 10. 实验结果应该怎样解读？

### 10.1 nuScenes-SurroundOcc

| 方法 | 模态 | IoU | mIoU |
|---|---|---:|---:|
| GaussianFormer | C | 29.8 | 19.1 |
| GaussianFormer-2 | C | 31.7 | 20.8 |
| Co-Occ | L+C | 41.1 | **27.1** |
| **GaussianFormer3D** | **L+C** | **43.3** | **27.1** |

主要结论：它相对 camera-only GaussianFormer 提升巨大，并在 IoU 上超过 Co-Occ；但 mIoU 与 Co-Occ 相同，并非所有指标都形成绝对领先。

### 10.2 nuScenes-Occ3D

GaussianFormer3D 报告 mIoU **46.4**，而 OccFusion 为 **48.7**。因此不能笼统地称它为所有 benchmark 上的 SOTA；它更明确的优势是 Gaussian 表示下的多模态融合、总体 occupancy IoU 和 memory efficiency。

### 10.3 RELLIS3D-WildOcc

GaussianFormer3D 报告：

$$
\text{IoU}=33.9,\qquad\text{mIoU}=13.1.
$$

M-OFFOcc(4-frame) 的结果约为 IoU 32.8、mIoU 14.8。GaussianFormer3D 总体 occupancy 更好，但类别平均性能较低。论文也指出极少出现的 barrier、rubble 和细微 terrain 类型依然困难。

这里的 “WildOcc” 是 off-road semantic classes，不等于 open-set/OOD detection。模型仍使用预定义闭集标签训练和评估。

### 10.4 恶劣条件下 LiDAR 的收益

| 条件 | GaussianFormer IoU | GaussianFormer3D IoU | 增益 |
|---|---:|---:|---:|
| Sunny | 29.6 | 43.6 | +14.0 |
| Rainy | 27.5 | 41.6 | +14.1 |
| Day | 30.3 | 43.6 | +13.3 |
| Night | 19.5 | 40.5 | **+21.0** |

夜间 IoU 提升最显著，符合 LiDAR 不依赖环境光、能稳定提供几何位置的预期。但夜间 mIoU 仍只有 15.5，说明准确几何不能完全补偿视觉语义退化。

### 10.5 Gaussian 数量、速度与内存

论文在单张 NVIDIA A40、batch size 1 上报告：

| Gaussians | Latency | 近似 FPS | Memory | IoU | mIoU |
|---:|---:|---:|---:|---:|---:|
| 6,400 | 415 ms | 2.41 | 4.9 GB | 39.6 | 21.4 |
| 12,800 | 462 ms | 2.16 | 5.0 GB | 41.4 | 24.2 |
| 25,600 | 555 ms | **1.80** | 5.5 GB | **43.3** | **27.1** |

它的优势是 memory，不是实时速度：

- Co-Occ：580 ms、11.8 GB、41.1/27.1；
- GaussianFormer3D：555 ms、5.5 GB、43.3/27.1。

因此“Gaussian 很稀疏，所以一定快”并不成立。25,600 queries × 4 blocks × 6 cameras × 4 scales × 多个 3D sampling points，仍然非常重。

---

## 11. 与 S2GO 的核心差异

| 维度 | GaussianFormer3D | S2GO |
|---|---|---|
| 正常 inference 输入 | Camera + LiDAR | Camera |
| LiDAR 作用 | 每帧 V2G 初始化 + depth-guided fusion | Stage-1 query pretraining 的几何老师 |
| query–Gaussian 关系 | 1 query ↔ 1 Gaussian | 1 parent query → 多个 child Gaussians |
| Gaussian 数量 | 25,600 | 每个 parent 解码一小簇 children |
| query 是否时序传播 | 否 | 是，Past/New/Future 角色 |
| learned temporal memory | 无 | 有 |
| velocity / future rollout | 无 | 有 |
| entity / instance | 无 | parent query 有局部 entity 雏形，但不是真 instance |
| 语义 | closed-set voxel supervision | closed-set occupancy + rendering/pretraining |
| 主要优势 | 强 LiDAR–camera 几何匹配、开源、低 memory | 层级表示、时序压缩、较高速度 |

最关键的结构区别可以画成：

```text
GaussianFormer3D

q1 ↔ G1
q2 ↔ G2
q3 ↔ G3        25,600 个同层 primitive
...


S2GO

parent q1 ──► G11, G12, ... G1J
parent q2 ──► G21, G22, ... G2J
parent q3 ──► G31, G32, ... G3J
      │
      └── parent query 才是被时序传播的压缩状态
```

所以，如果我们直接采用 GaussianFormer3D，不能说已经有 entity layer；如果采用 S2GO，也不能说 parent query 已经自动等于 object instance。

---

## 12. 从 OOD occupancy 角度看，它做对了什么？

### 12.1 对未知物体友好的部分

#### 几何初始化不依赖已知类别

V2G 只要求某处有 LiDAR return，不要求它属于 car、pedestrian 等已知类。未知障碍物也可以产生 Gaussian seeds。这对 class-agnostic occupied-space recall 很有价值。

#### 视觉 feature 与准确深度被匹配到同一 primitive

未知物体的视觉外观可能超出 closed-set classifier，但 foundation visual feature 仍可能包含有用的形状、材质或文本对齐信息。DFA 提供了把这些视觉 feature 放到正确 3D 深度的方法。

#### 连续 Gaussian 适合把 entity 形状再投回 voxel

最终任务仍要求 unknown occupancy mask。Gaussian 天然能作为 entity 与 voxel 之间的中间表示：entity 管高层判定，Gaussians 管局部形状，G2V 管最终栅格输出。

#### LiDAR 提升夜间几何稳定性

OOD safety case 往往出现在夜间、弱光和恶劣天气。论文中的夜间 IoU 增益说明 geometry branch 很值得保留。

### 12.2 它尚未解决的 OOD 问题

#### 仍是 closed-set semantic head

每个 Gaussian 只预测固定的 17 类前景 semantic coefficients，没有：

- unknown / novel head；
- open-vocabulary embedding；
- energy / density / evidence score；
- known-vs-unknown calibration；
- OOD training samples 或 outlier exposure。

未知物体最终仍会被吸进某个已知类别，或者变成低置信度但未被显式识别的 occupied voxels。

#### Gaussian 是 primitive，不是 entity

OOD 判断不适合只在独立 voxel/Gaussian 上做，因为单个小 primitive 缺少完整物体上下文。当前模型没有把空间上属于同一物体的 Gaussians 聚合成 entity，也没有统一的 objectness、novelty 和风险评分。

#### LiDAR depth gate 会漏掉“图像看见、LiDAR 没打到”的新物体

对于远距离小障碍物、低反射材质、稀疏 beam 之间的物体，camera 可能有强视觉证据，但 sparse depth support 为零。纯乘法 lifting 会抑制这条证据。

#### 随机 subsampling 不关心 novel evidence

当非空 voxels 超过 25,600 时，V2G 是随机采样。大片道路、建筑和植被可能消耗 query budget，小型未知障碍不一定被保留。

#### 多 sweep 可能产生动态拖影

新奇动态物体尤其需要精确的时序身份和运动补偿。简单叠加 10 个 sweeps 可能把移动物体拉长，而模型没有 velocity 或 persistent entity 去解释这种变化。

#### 没有 observation-evidence 分解

当前 opacity 同时承担 Gaussian contribution，并以 LiDAR intensity 初始化。它没有明确区分：

- 这个空间是否被传感器实际观察；
- 是否有 occupied geometry evidence；
- 是否能给出已知语义；
- 是否因遮挡/距离而未知。

对 OOD 来说，“看到了一个未知物体”和“这里根本没被可靠观测”必须分开。

---

## 13. 我建议怎样把它改成我们的模型？

### 13.1 推荐总体结构

```text
       Camera foundation features                     LiDAR
       DINO / SAM-family / driving pretrain             │
                    │                                    ▼
                    │                         geometry-aware V2G seeds
                    │                         xyz, local shape, return age
                    │                                    │
                    ├──────────────┬─────────────────────┘
                    │              │
                    ▼              ▼
          camera-only 3D evidence   LiDAR-guided (u,v,d) attention
                    │              │
                    └────── gated asymmetric fusion ─────┘
                                   │
                          child Gaussian tokens
                                   │
                    spatial grouping / entity decoder
                                   │
             ┌─────────────────────┼──────────────────────┐
             ▼                     ▼                      ▼
      geometry/existence       open semantic          observation
      center/scale/opacity     embedding + known/OOD  evidence/visibility
             │                     │                      │
             └──────────── persistent entity memory ─────┘
                                   │
                      entity-conditioned Gaussians
                                   │
                                   ▼
                  known + unknown Gaussian-to-voxel
```

### 13.2 保留 GaussianFormer3D 的三个模块

#### 保留 V2G，但改成 OOD-aware seed selection

不要随机抽非空 voxels。建议按三类 budget 采样：

1. **coverage seeds**：FPS / voxel-stratified sampling，保证空间覆盖；
2. **visual residual seeds**：图像出现强 foreground / anomaly，但现有 Gaussians 未解释的区域；
3. **geometric residual seeds**：LiDAR returns 与当前 known occupancy prediction 不一致的区域。

可以设计 seed priority：

$$
s_v=\lambda_1s_{\text{coverage}}
+\lambda_2s_{\text{visual anomaly}}
+\lambda_3s_{\text{unexplained LiDAR}}
+\lambda_4s_{\text{risk}}.
$$

这才真正接近 OOD object proposal，而原版 V2G 只是 surface primitive proposal。

#### 保留 $(u,v,d)$ deformable attention

这是 GaussianFormer3D 最值得借用的融合模块。它很好地回答了我们前面提出的问题：

> 先有 Gaussian / visual hypothesis，再从 LiDAR 提供的 depth-supported feature 中取几何一致的视觉信息。

但建议增加 camera-only fallback：

$$
f_i
=f^{\text{cam-only}}_i
+g_i\odot f^{\text{LiDAR-guided}}_i,
$$

其中门控 $g_i$ 由 LiDAR coverage、投影误差、距离和传感器可靠性共同决定。这样 LiDAR 有 return 时负责精确深度；没有 return 时，模型仍能保留 camera 的潜在 OOD evidence。

#### 保留 G2V

最终 benchmark 和规划模块通常仍要 dense BEV occupancy。G2V 是把稀疏 entity/Gaussian representation 转回 voxel prediction 的自然接口。

### 13.3 必须新增 parent/entity 层

建议在 child Gaussians 上方引入 $K$ 个 entity queries：

$$
e_k=\operatorname{Aggregate}\{q_i:w_{ki}>0\},
$$

$$
w_{ki}=\operatorname{softassign}(e_k,q_i,m_i).
$$

每个 entity 预测：

- class-agnostic existence；
- center、extent 或 mask；
- open-semantic embedding；
- known-class logits；
- OOD / novelty score；
- observation evidence；
- velocity 与 temporal identity。

然后将 entity 结论广播回 child Gaussians，再 splat 成 unknown occupancy：

$$
c_i^{\text{unknown}}
=\sum_k w_{ki}\,p_k^{\text{OOD}}.
$$

这样，是否 unknown 是基于“一整块空间/一个实体”的上下文判断，但最终几何边界仍由多个 Gaussian / voxels 表示。

### 13.4 语义和几何要做职责分离

推荐把 feature 拆成：

$$
q_i=[q_i^{\text{sem}},q_i^{\text{geo}},q_i^{\text{obs}}].
$$

- Camera foundation model 主要更新 $q^{\text{sem}}$；
- LiDAR/V2G 主要更新 $q^{\text{geo}}$ 和 $q^{\text{obs}}$；
- shared transformer 允许有限的信息交换；
- semantic/OOD head 不直接使用原始 LiDAR intensity shortcut；
- geometry head 不依赖 closed-set semantic classifier 才能判断 occupied。

这比简单说“camera attention weight 更大”更可控。

### 13.5 加入 S2GO 式 temporal parent memory，而不是传播 25,600 个 Gaussians

每帧传播所有 Gaussian 太慢，也容易产生不稳定 identity。更合理的是：

```text
current child Gaussians ──► aggregate to entity queries
                                   │
                            select persistent entities
                                   │ ego + motion align
                                   ▼
                           next-frame entity memory
```

尤其应给 OOD entity 设置 risk-aware retention：高 novelty 但暂时低置信度的 entity，至少保留 2–3 帧积累证据，避免被 top-opacity 策略立刻淘汰。

---

## 14. 推荐的研究实施顺序

### Phase A：复现 closed-set baseline

目标是确认官方结果、吞吐和内存，不立刻改 OOD。

- 跑通官方 nuScenes-SurroundOcc checkpoint；
- 可视化 LiDAR-initialized Gaussians、最终 Gaussians 和 occupancy；
- 测量 6.4k / 12.8k / 25.6k query latency；
- 将 covariance inverse 改成 GPU-safe 实现后重新 benchmark；
- 单独验证 $N_g>N_v$ 的 sparse-input V2G 分支。

### Phase B：确认“视觉主导、LiDAR 几何”是否成立

做四组 controlled ablation：

1. Camera-only GaussianFormer-like baseline；
2. V2G only；
3. LiDAR-guided DFA only；
4. V2G + DFA。

再加入 modality corruption：

- 降低 LiDAR beam / 随机 drop returns；
- LiDAR–camera calibration perturbation；
- image blur / night / rain；
- one sweep vs ten sweeps；
- dynamic-object motion trail。

不仅看 mIoU，还要看各类 feature 与预测对某个模态的敏感性。

### Phase C：加入 open-semantic Gaussian head

给每个 Gaussian 或 entity 输出 normalized visual embedding：

$$
z_i=\frac{Wq_i^{\text{sem}}}{\|Wq_i^{\text{sem}}\|_2},
$$

与文本/视觉 prototype 对齐，同时保留 class-agnostic occupancy head。先建立 voxel-level OOD baseline：MSP、energy、prototype distance、evidential score，再比较 entity aggregation 前后差异。

### Phase D：加入 entity grouping

评估：

- entity recall；
- unknown object recall；
- entity fragmentation / merging；
- voxel AUROC、AUPR、FPR95；
- known-class mIoU；
- near-field risk-weighted OOD recall。

### Phase E：加入 temporal memory

先只传播 entity state，不传播完整 voxel grid 或 25,600 Gaussian bank。比较：

- 无时序；
- multi-sweep only；
- persistent entity memory；
- entity memory + residual query birth。

这样能分清“叠加历史点云的几何收益”和“学习到的 OOD 证据积累”不是同一件事。

---

## 15. 复现与工程注意事项

### 15.1 官方已提供什么？

截至核对日期，官方仓库已经公开：

- nuScenes-SurroundOcc 与 RELLIS3D-WildOcc 配置；
- train / eval / visualization 脚本；
- 两个 benchmark 的 pretrained checkpoints；
- V2G、Gaussian encoder、G2V 源码；
- 3D deformable attention 和 G2V 所需的 custom CUDA build instructions；
- 预生成 nuScenes depth maps 与 data-info pkl 下载链接。

这使它比尚未开源的 S2GO 更适合作为近期实验起点。

### 15.2 环境

官方给出两组主要组合：

- Python 3.8.16；
- PyTorch 2.0.0 + CUDA 11.8，或 PyTorch 2.1.0 + CUDA 12.1；
- MMCV 2.0/2.1；
- MMDetection 3.x；
- MMSegmentation 1.x；
- MMDetection3D 1.1/1.3；
- spconv；
- DFA3D custom CUDA extension；
- Gaussian-to-voxel custom CUDA extension。

官方训练命令使用 8 张 GPU；论文写 nuScenes global batch size 8、A40 GPU。

### 15.3 代码成熟度与许可证

公开仓库当前历史很短，顶层没有看到明确 LICENSE 文件。部分 G2V 文件保留了 Graphdeco 的“non-commercial, research and evaluation use”声明，并引用一个仓库中不存在的 `LICENSE.md`。

因此：

- 科研复现实验通常可先进行；
- 若要大范围重发布、商业使用或将代码合并进正式产品，必须向作者确认整体授权和第三方 CUDA code 的许可证；
- 我对源码做了 Python syntax compile 检查，能够通过；这不等于 custom CUDA ops、数据接口和完整 checkpoint 已在本机端到端运行。

---

## 16. 哪些东西值得直接借，哪些不要照搬？

### 建议直接借用

1. **LiDAR voxel → Gaussian geometry seeds** 的接口；
2. **Gaussian-aligned 3D keypoints**；
3. **投影前 + 投影后两阶段 offsets**；
4. **用 LiDAR sparse depth 解决 image feature 深度歧义**；
5. **局部 G2V aggregation**；
6. **只让一个大 Gaussian 表示 empty space** 的效率思想；
7. **连续 Gaussian 支持多分辨率 occupancy 输出**。

### 不建议原样照搬

1. 随机选择 25,600 个非空 voxels；
2. 用 intensity 直接初始化 opacity，而不给独立 observation confidence；
3. 只有 LiDAR depth support 才允许视觉 feature 进入 3D；
4. 17 类 closed-set softplus semantic coefficients；
5. one query ↔ one Gaussian 的无层级表示；
6. 没有 object/entity aggregation；
7. 10-sweep 简单叠加而无动态补偿；
8. 对所有 camera 直接求平均而默认不使用 visibility；
9. CPU inverse 再 `.cuda()` 的 G2V 实现；
10. 555 ms/frame 的主配置直接作为在线系统。

---

## 17. 最终评价

### 论文贡献是否扎实？

是。两个模块都瞄准了 camera-only Gaussian occupancy 的真实瓶颈，而且单独消融和组合消融都很强。特别是它没有把 LiDAR 只是 concat 到 image feature，而是把 LiDAR 用到：

- primitive initialization；
- depth-aware 3D feature matching；
- 迭代 Gaussian refinement。

这比传统 late fusion 更符合我们讨论的现代 feature-level asymmetric fusion。

### 它是不是目前我们想找的“LiDAR 给 queries、Gaussian 做 occupancy”的工作？

是，而且比“用 LiDAR 只做训练期 query teacher”的 S2GO 更直接：

> **GaussianFormer3D 在每次 inference 时，用 LiDAR non-empty voxels 真正初始化 Gaussian/query 的空间位置，并继续用 LiDAR depth 指导视觉 feature sampling。**

### 它能不能直接解决 OOD object occupancy？

不能。它提供了优秀的几何和融合 backbone，但没有 OOD 建模。直接加一个 unknown logit 可能得到 baseline，却很难解决 entity consistency、未观测区域、视觉开放语义和跨帧证据积累。

### 对我们最合适的定位

最准确的项目表述应当是：

> **We build upon the LiDAR-guided Gaussian occupancy backbone of GaussianFormer3D, and introduce camera-dominant open-semantic features, OOD-aware query birth, hierarchical Gaussian entities, and temporal evidence accumulation for open-set 3D occupancy prediction.**

中文：

> **我们以 GaussianFormer3D 的 LiDAR 引导 Gaussian occupancy 骨架为基础，新增视觉主导的开放语义、面向未知物体的 query birth、层级 Gaussian entity 表示，以及跨帧观测证据积累，从而进行 open-set 3D occupancy prediction。**

这比说“基于 GaussianFormer3D 加一个 OOD head”更准确，也更能形成独立的科研贡献。

---

## 18. 原始资料

- [GaussianFormer3D 论文（arXiv）](https://arxiv.org/abs/2505.10685)
- [GaussianFormer3D 官方项目页](https://lunarlab-gatech.github.io/GaussianFormer3D/)
- [GaussianFormer3D 官方代码](https://github.com/lunarlab-gatech/GaussianFormer3D)
- [GaussianFormer](https://github.com/huang-yh/GaussianFormer)
- [DFA3D / 3D Deformable Attention](https://github.com/IDEA-Research/3D-deformable-attention)

本文中的模型结构、超参数、消融和 benchmark 数字优先以论文为准；关于测试 pipeline、`DepthHead_GTDpt`、V2G 分支、query/Gaussian 对应、G2V CUDA 路径及工程风险的结论来自公开代码核对。
