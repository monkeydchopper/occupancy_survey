# SAM 3D Objects 详解：如何从 Object Mask 生成 3D Gaussian，以及怎样用于 OOD Occupancy

> 研究对象：Meta **SAM 3D Objects / 3Dfy Anything in Images**（2025）  
> 论文：[arXiv:2511.16624](https://arxiv.org/abs/2511.16624)  
> 官方介绍：[Meta AI Research](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)  
> 官方代码：[facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)  
> 本地代码：`/home/yunxuan/code/opensource/sam-3d-objects`

---

## 0. 先说结论

SAM 3D Objects 对我们的 OOD occupancy 很有价值，但正确用法不应该是：

> 在自动驾驶的每一帧中，对每一个 object proposal 完整运行一次 SAM 3D，再把它输出的几十万个 Gaussian 塞进 occupancy。

更合理的用法是：

> **使用 SAM 3D 作为 object-level generative 3D prior，在训练阶段指导视觉主导的 entity query 学会生成紧凑的 occupancy Gaussians；LiDAR 负责公制位置、可见表面和自由空间约束。**

SAM 3D 最值得借用的不是它最终输出的带颜色 3D asset，而是下面这条建模链路：

```text
object mask + object appearance
              │
              ▼
     object-level representation
              │
              ▼
       canonical 3D shape
              │
              ▼
        metric 3D layout
              │
              ▼
        Gaussian object
```

这和我们设想的 **entity query → child Gaussian bundle** 非常接近。

---

## 1. 先避免名字混淆

本文讨论的是 Meta 在 2025 年发布的 **SAM 3D Objects**：输入一张 RGB 图像和一个选中的物体 mask，生成完整的三维物体。

它不是早期同样被叫作 SAM3D 的点云分割、医疗分割或者“将 SAM mask 投影到点云”的方法。

在交互 Demo 中，用户看到的过程通常是：

```text
点击或框选物体
      │
      ▼
SAM / SAM 3 产生 2D mask
      │
      ▼
SAM 3D Objects 接收 image + mask
      │
      ▼
生成一个由大量 Gaussian splats 构成的 3D object
```

所以准确地说：

- SAM 或 SAM 3 负责“把哪个物体选出来”；
- SAM 3D Objects 负责“根据选中的图像区域生成三维物体”。

SAM 3D 本身并不要求 mask 一定由某个特定版本的 SAM 产生，也可以使用人工 mask、数据集 instance mask 或我们自己的 entity segmentation head。

---

## 2. 它解决的任务是什么？

输入：

- 一张 RGB 图像 $I$；
- 一个二值 object mask $M$；
- 可选的 point map $P$。

输出：

- 完整物体形状 $S$；
- 纹理 $T$；
- 物体在相机坐标系中的 rotation $R$；
- translation $t$；
- scale $s$。

可以写成：

$$
p(S,T,R,t,s\mid I,M,P).
$$

这里是一个**生成分布**，而不是简单的确定性回归。原因是单张图像本身无法唯一决定被遮挡的物体背面：模型只能依据大规模三维数据中学到的先验，生成一个合理的三维解释。

因此，SAM 3D 的完整输出应该理解为：

> 与当前图像和 mask 相容的、具有较高概率的完整三维物体假设。

它不能被直接理解为：

> 传感器已经观测并确认的完整三维几何。

这个区别对普通 3D asset generation 没那么重要，但对安全相关的 occupancy 至关重要。

---

## 3. 整体模型：不是 mask 直接回归几个 Gaussian

SAM 3D Objects 的主要数据流是：

```text
RGB image + selected object mask + optional point map
                         │
                         ▼
                  DINOv2 condition encoder
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Geometry and Layout Generator                 │
│                                                        │
│ 生成：                                                  │
│ · coarse canonical shape O                            │
│ · rotation R                                          │
│ · translation t                                       │
│ · scale s                                             │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
              找出 coarse shape 的 active voxels
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Sparse Latent Texture/Geometry Refinement     │
│                                                        │
│ active voxels + image conditions → structured latent   │
│ 即 SLAT                                                │
└────────────────────────────────────────────────────────┘
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
          Mesh decoder       Gaussian decoder
                                   │
                                   ▼
                       大量带颜色的 Gaussian splats
```

关键点是：

1. 它先生成 coarse shape 和 layout；
2. 再生成稀疏的 structured latent；
3. 最后才用专门的 decoder 解码为 mesh 或 Gaussian；
4. 强大的补全能力主要来自大规模三维生成先验和两阶段 flow model，而不是 mask 本身。

本地端到端 inference pipeline 位于：

- [`inference_pipeline.py`](/home/yunxuan/code/opensource/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline.py)

---

## 4. 输入图像和 mask 是怎样编码的？

模型并不是只裁剪 mask 内部的 RGB。它会构造四组视觉条件：

1. cropped object RGB；
2. cropped binary mask；
3. full-image RGB；
4. full-image mask。

直觉如下：

```text
Object Crop                             Full Image
┌─────────────────┐                    ┌──────────────────────┐
│  高分辨率物体细节 │                    │  场景上下文和物体关系   │
│  局部形状、纹理   │                    │  类别线索、遮挡线索     │
└─────────────────┘                    └──────────────────────┘
          │                                        │
          └────────────────┬───────────────────────┘
                           ▼
                    DINOv2 features
```

crop 可以最大限度保留物体的局部细节；full image 则帮助模型判断物体类别、场景环境和遮挡关系。例如只露出一部分轮子时，完整场景可以帮助判断它属于汽车、自行车还是其他结构。

官方 released configuration 使用 DINOv2 ViT-L/14 Registers 作为视觉条件编码器。

### 4.1 可选 point map

SAM 3D 还可以接收一个 point map：

$$
P(u,v)=(X(u,v),Y(u,v),Z(u,v)).
$$

它表示像素 $(u,v)$ 对应的三维点。

point map 可以来自：

- 具有深度能力的硬件传感器；
- 单目深度/几何模型；
- 投影到相机平面的车载 LiDAR。

官方配置在没有外部 point map 时，可以使用 MoGe 预测单目 point map。对我们的 camera-LiDAR 系统而言，更重要的是使用标定参数将 LiDAR 点投影到每个相机：

$$
\tilde p_{mathrm{cam}} = K T_{mathrm{LiDAR}\rightarrow\mathrm{cam}}p_{mathrm{LiDAR}}.
$$

然后只保留落在 entity mask 内的有效深度点，形成稀疏的公制 point map。

本地 point-map pipeline 位于：

- [`inference_pipeline_pointmap.py`](/home/yunxuan/code/opensource/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline_pointmap.py)

需要注意：车载 LiDAR 投影通常比手机深度或渲染得到的 dense point map 稀疏得多，因此不能假设官方 point-map encoder 可以不经适配直接工作。后续需要提供：

- point validity mask；
- sparse-depth encoder；
- 或者具有置信度的 depth completion。

---

## 5. Stage 1：联合生成 coarse shape 和 3D layout

第一阶段联合预测：

$$
(O,R,t,s),
$$

其中：

| 变量 | 含义 | 作用 |
|---|---|---|
| $O$ | canonical object space 中的粗形状 | 表示物体自身几何 |
| $R$ | rotation | 表示物体朝向 |
| $t$ | translation | 表示相机坐标中的位置 |
| $s$ | anisotropic scale | 表示物体三个方向的尺寸 |

coarse shape 使用 $64^3$ 的空间分辨率。其内部 shape latent 大约是：

$$
16^3\times 8,
$$

也就是 4096 个 shape tokens；rotation、translation 和 scale 各有自己的 layout token。

### 5.1 为什么 shape 和 layout 要一起生成？

物体形状和空间姿态不是完全独立的。例如：

- 已知车轮在图像中的位置，有助于推断汽车的朝向；
- 已知物体可能是一辆公交车，有助于判断其真实尺寸和距离；
- point map 给出的少量三维点可以帮助确定 scale 和 translation；
- layout 反过来又能帮助模型解释图像中的投影形状。

因此模型使用独立但会交换信息的 shape/layout transformer streams，而不是先完全独立地生成物体，再做一个不相关的 pose regression。

### 5.2 它是生成而不是一次性回归

Stage 1 使用 conditional rectified flow matching。可以把它理解为从随机噪声开始，逐步生成符合图像条件的 shape 和 layout：

$$
z_0\sim\mathcal N(0,I)
\quad\longrightarrow\quad
z_1=(O,R,t,s).
$$

这意味着同一个单视图输入可以存在多个合理的三维结果。模型对不可见区域做的是概率性补全，不是从图像中恢复出了不存在的观测信息。

---

## 6. Stage 2：从 active voxels 生成 SLAT

第一阶段得到 $64^3$ coarse shape 后，模型找到其中的 active voxels。第二阶段只在这些稀疏位置上生成 structured latent，因此不需要在整个立方体中维护高分辨率 dense features。

可以抽象为：

$$
Z_{mathrm{SLAT}}
=
F_{mathrm{SLAT}}
\left(
\mathcal V_{mathrm{active}},
F_{mathrm{image}},
F_{mathrm{mask}}
\right).
$$

其中：

- $\mathcal V_{mathrm{active}}$ 是 coarse shape 中的活跃体素集合；
- $F_{mathrm{image}}$ 和 $F_{mathrm{mask}}$ 是视觉条件；
- $Z_{mathrm{SLAT}}$ 是每个 active voxel 对应的 structured latent。

同一套 SLAT 可以通过不同 decoder 输出：

- mesh；
- 3D Gaussian splats。

因此，模型不是用 mesh 转 Gaussian，也不是对渲染得到的点云做 Gaussian fitting，而是直接从共享的三维 structured latent 解码不同表示。

---

## 7. Gaussian 到底是怎样生成的？

### 7.1 一个 active voxel 生成多个 Gaussian

假设 $v$ 是一个 active voxel，Gaussian decoder 会为它产生多个 child Gaussians：

$$
G_{v,j}
=
\left(
x_v+\Delta x_{v,j},
c_{v,j},
s_{v,j},
q_{v,j},
\alpha_{v,j}
\right).
$$

各参数含义如下：

| 参数 | 含义 |
|---|---|
| $x_v+\Delta x_{v,j}$ | active voxel 附近的 Gaussian center |
| $c_{v,j}$ | RGB/DC color feature |
| $s_{v,j}$ | 三轴 anisotropic scale |
| $q_{v,j}$ | rotation quaternion |
| $\alpha_{v,j}$ | opacity |

默认 decoder 可以为每个 active voxel 生成最多 32 个 Gaussian；代码中也存在每个 voxel 生成 4 个 Gaussian 的变体。

本地 Gaussian decoder：

- [`decoder_gs.py`](/home/yunxuan/code/opensource/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/decoder_gs.py)

### 7.2 为什么一个 voxel 里需要多个 Gaussian？

一个 Gaussian 只能表示一个具有中心、朝向、尺度和透明度的椭球。复杂表面可能同时包含：

- 多个不同方向的局部平面；
- 薄结构；
- 锐利边缘；
- 颜色变化；
- 高曲率几何。

因此 decoder 在一个 active voxel 内布置多个 Gaussian，使它们共同逼近复杂的局部表面。

### 7.3 它输出的不是“一个 Gaussian”

用户在 Demo 中看到的是一个完整的 Gaussian object，但这个 object 由大量 splats 构成。

我们检查了本地生成的两个 `.ply` 文件：

- `outputs/codex_splat.ply`；
- `outputs/kidsroom_obj14_splat.ply`。

两者都包含约 842,464 个 Gaussian vertices。若按照每个 active voxel 32 个 Gaussian 计算，对应约 26,327 个 active voxels。

所以它与 S2GO 的计算量级差别很大：

| 方法 | 中间父节点 | 每个父节点的 child Gaussians | 目标 |
|---|---|---:|---|
| S2GO | spatial query | 约 10～20 | 紧凑 occupancy 表示 |
| SAM 3D Objects | active voxel / SLAT | 默认最多 32 | 高质量可渲染 3D asset |
| 我们希望设计的模型 | object/entity query | 约 10～64 | object-level OOD occupancy |

---

## 8. Gaussian 怎样放回相机和车体坐标系？

Gaussian 最初位于 normalized canonical object space，可以近似理解为：

$$
[-0.5,0.5]^3.
$$

Stage 1 预测的 layout 将其放到相机坐标系：

$$
\mu_{mathrm{cam}}
=
R\operatorname{diag}(s)\mu_{mathrm{local}}+t.
$$

再利用相机外参变换到 ego 坐标系：

$$
\mu_{mathrm{ego}}
=
R_{mathrm{cam}\rightarrow\mathrm{ego}}
\mu_{mathrm{cam}}
+
t_{mathrm{cam}\rightarrow\mathrm{ego}}.
$$

如果 Gaussian 在 local space 中的 covariance 为 $\Sigma_{mathrm{local}}$，那么还需要相应地变换 covariance：

$$
A=R\operatorname{diag}(s),
$$

$$
\Sigma_{mathrm{cam}}
=
A\Sigma_{mathrm{local}}A^\top,
$$

$$
\Sigma_{mathrm{ego}}
=
R_{mathrm{cam}\rightarrow\mathrm{ego}}
\Sigma_{mathrm{cam}}
R_{\mathrm{cam}\rightarrow\mathrm{ego}}^\top.
$$

不能只变换 Gaussian center，而保持 orientation 和 scale 不变。

---

## 9. SAM 3D 为什么能补出不可见的背面？

不是因为它从正面图像中“看见”了背面，而是因为它从大量三维物体中学习了：

$$
p(\text{complete 3D shape}\mid\text{visible appearance, mask, context}).
$$

其训练包含大规模 synthetic pretraining、semi-synthetic mid-training 和真实数据 post-training。论文报告的预训练资产和样本量包括：

- 约 2.7M 个 object meshes；
- 每个 mesh 从多个视角渲染，得到约 64.8M 个训练样本；
- mid-training 使用约 61M 个 semi-synthetic render-and-paste samples；
- 之后再使用真实图像、人类偏好和 model-in-the-loop 数据进行 post-training。

这种数据规模使模型具有很强的 object-level 3D prior。对于常见物体，它能根据少量可见部分生成非常合理的完整形状。

但这种能力也有两面性：

```text
遮挡的常见物体
      │
      ▼
强生成先验 → 合理补全                 有利

真正陌生的 OOD 物体
      │
      ▼
强生成先验 → 强行解释成熟悉物体        有风险
```

后者可以称为 **familiarization bias**：模型可能把真正的新奇物体生成成最接近的已知三维形状，从而掩盖 novelty。

---

## 10. 为什么它适合启发 OOD Occupancy？

目前我们的 entity 建模有一个核心问题：

> 当 entity query 已经判断空间中“有一个东西”以后，它应该怎样生成完整、连续、物体级的三维形状？

SAM 3D 提供了一个强大的 object-centric 答案：

```text
2D object evidence
        │
        ▼
object/entity representation
        │
        ▼
canonical shape + layout
        │
        ▼
child Gaussian bundle
```

它对 OOD occupancy 可能带来的价值包括：

1. **从 voxel-level 判断提升到 entity-level 聚合**  
   一个物体的多块区域共享整体外观、形状、位置和 novelty 判断。

2. **遮挡补全能力**  
   对只被观察到一部分的物体，可以给出完整形状假设。

3. **长尾三维先验**  
   它见过的三维资产远多于典型自动驾驶 occupancy 数据集中的类别和实例。

4. **天然兼容 Gaussian 表示**  
   可以启发我们的 entity-to-Gaussian decoder，而不必始终回到 dense voxel features。

5. **支持 point-map condition**  
   LiDAR 可以自然地作为公制几何条件接入，而 camera 仍然负责发现物体和理解外观。

---

## 11. 为什么不能直接把完整 SAM 3D 串到自动驾驶模型后面？

### 11.1 计算和显存成本太高

官方安装建议至少约 32 GB GPU memory。released model 包含约 1.2B 参数的 geometry/layout generator 和约 600M 参数的 sparse latent generator。

它面向单物体高质量 3D asset generation，不是面向多相机、几十个物体、20 Hz 自动驾驶在线推理。

### 11.2 Gaussian 的字段不适合直接做 occupancy

SAM 3D 的 Gaussian 主要是：

$$
G^{\mathrm{SAM3D}}
=
(\mu,\Sigma,\alpha,c^{\mathrm{RGB}}).
$$

我们需要的是：

$$
G^{\mathrm{Occ}}
=
(\mu,\Sigma,p_{\mathrm{occ}},f_{\mathrm{semantic}},
p_{\mathrm{OOD}},u_{\mathrm{completion}}).
$$

它没有直接提供：

- semantic occupancy logits；
- CLIP-aligned open-vocabulary feature；
- OOD probability；
- 观测证据；
- 补全不确定性；
- motion/velocity；
- temporal identity。

### 11.3 单物体、单图像，不保证多视角和时序一致

released architecture 的基本单位是单张图像中的一个物体。自动驾驶则需要：

- 同一物体跨多个相机的一致 entity；
- 跨时间的 persistent identity；
- ego-motion alignment；
- 不同物体之间不发生不合理 interpenetration；
- 对 road、wall、vegetation 等 stuff 区域进行场景级建模。

### 11.4 生成式补全不能等价于观测事实

SAM 3D 生成的不可见背面可能很合理，但依然是 prior，而不是 evidence。如果直接写入 occupancy，会让规划模块无法区分“确实观测到”和“模型猜测存在”。

---

## 12. 推荐架构：Scene Branch + Entity Branch

我建议把整个系统拆成两个互补分支：

```text
Multi-view Cameras                              LiDAR
        │                                         │
        ▼                                         ▼
large pretrained visual encoder             LiDAR encoder
        │                                         │
        └──── camera-query cross attention ◄──────┘
                          │
                          ▼
                  shared BEV/3D features
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Scene / Stuff Branch          Entity Branch
      road, wall, terrain           objects / OOD objects
      dense/sparse Gaussians              │
                                          ▼
                               persistent entity query
                                          │
                                          ▼
                              compact Gaussian decoder
                                          │
                      ┌───────────────────┼───────────────────┐
                      ▼                   ▼                   ▼
              observed geometry     shape completion     semantics/OOD
```

### 12.1 Scene / Stuff Branch

负责：

- road；
- sidewalk；
- wall；
- terrain；
- vegetation；
- 其他不适合被拆成 object instance 的场景区域。

这一分支可以采用 GaussianFormer3D 类的场景级 Gaussian occupancy，或者普通 voxel/BEV occupancy decoder。

### 12.2 Entity Branch

负责：

- objectness；
- instance mask；
- persistent identity；
- object-level semantics；
- OOD score；
- canonical shape；
- 3D layout；
- child Gaussian bundle。

一个 entity query 可以写成：

$$
e_k=
\left[
f_k^{\mathrm{visual}},
g_k^{\mathrm{LiDAR}},
R_k,t_k,s_k,
z_k^{\mathrm{shape}},
h_k^{\mathrm{temporal}}
\right].
$$

其中 camera 是 entity existence、外观和语义的主要来源；LiDAR 主要提供 metric geometry。

---

## 13. 怎样让 camera 主导、LiDAR 提供几何？

我们之前提出的非对称 cross-attention 很适合这里：

$$
Q=F_{mathrm{entity}}^{\mathrm{camera}},
\qquad
K,V=F^{\mathrm{LiDAR}}.
$$

$$
F_{mathrm{entity}}^{\mathrm{fused}}
=
F_{mathrm{entity}}^{\mathrm{camera}}
+
\operatorname{CrossAttn}
\left(
Q_{mathrm{camera}},K_{mathrm{LiDAR}},V_{mathrm{LiDAR}}
\right).
$$

这意味着：

- camera query 决定“要查询哪个物体”；
- LiDAR 不是独立发起物体 proposal 的主模态；
- camera 从相应空间位置抽取 LiDAR geometry；
- LiDAR absence 不能轻易否定 camera 中明确出现的物体；
- LiDAR evidence 可以修正深度、尺寸、朝向和表面位置。

可以将职责划分为：

| 能力 | 主要模态 |
|---|---|
| 是否存在一个 object/entity | Camera |
| 外观、类别、open-vocabulary feature | Camera |
| OOD/novelty 判断 | 以 Camera 为主，融合几何冲突 |
| 绝对深度和 metric translation | LiDAR |
| 物体尺寸和表面位置 | Camera + LiDAR |
| LiDAR ray free space | LiDAR |
| 不可见区域形状补全 | SAM 3D-like prior |

---

## 14. 推荐接法：SAM 3D 作为训练阶段 Teacher

这是最现实、也最有研究价值的方案。

### 14.1 Teacher 输入

对于训练数据中的每个 entity：

1. 使用 GT instance mask、SAM mask 或我们的 entity mask；
2. 从最佳相机视角裁剪 object RGB 和 mask；
3. 将 LiDAR 投影到该相机；
4. 构造带 validity/confidence 的 metric point map；
5. 将 image、mask、point map 输入冻结的 SAM 3D teacher。

Teacher 产生：

$$
(O_T,R_T,t_T,s_T,Z_T^{\mathrm{SLAT}}),
$$

以及可选的 Gaussian object。

### 14.2 Student 输出

我们的轻量 student entity query 预测：

$$
(O_S,R_S,t_S,s_S,z_S),
$$

再生成少量 occupancy child Gaussians：

$$
e_k
\longrightarrow
\{G_{k,1},\ldots,G_{k,J}\},
\qquad J\approx 10\text{--}64.
$$

每个 child Gaussian 包含：

$$
G_{k,j}
=
\left(
\mu_{k,j},
\Sigma_{k,j},
\alpha_{k,j},
f_{k,j}^{\mathrm{semantic}},
p_{k,j}^{\mathrm{OOD}},
u_{k,j}
\right).
$$

### 14.3 不要直接匹配几十万个 Gaussian

SAM 3D 输出的 Gaussians 数量很大，而且是无序集合。直接做逐 Gaussian matching 会遇到：

- 数量不一致；
- permutation ambiguity；
- student 不需要保留渲染纹理细节；
- 巨大的训练显存和计算成本。

更合理的蒸馏目标是：

1. $64^3$ canonical coarse occupancy；
2. active-voxel probability；
3. 经过 adapter 压缩的 SLAT；
4. 从多个视角渲染出的 silhouette/depth；
5. 经过 Gaussian compression 后的少量 shape primitives。

一种基本的 teacher-student loss 是：

$$
\mathcal L_{\mathrm{prior}}
=
\mathcal L_{\mathrm{shape}}
\left(O_S,\operatorname{sg}(O_T)\right)
+
\mathcal L_{\mathrm{layout}}
\left((R_S,t_S,s_S),(R_T,t_T,s_T)\right).
$$

其中 $\operatorname{sg}$ 表示 stop gradient。

整体训练目标可以写成：

$$
\begin{aligned}
\mathcal L={}&
\lambda_{\mathrm{prior}}\mathcal L_{\mathrm{shape\text{-}distill}}
+\lambda_{\mathrm{lidar}}\mathcal L_{\mathrm{surface}}\\
&+\lambda_{\mathrm{free}}\mathcal L_{\mathrm{free\text{-}space}}
+\lambda_{\mathrm{mask}}\mathcal L_{\mathrm{render\text{-}mask}}\\
&+\lambda_{\mathrm{depth}}\mathcal L_{\mathrm{render\text{-}depth}}
+\lambda_{\mathrm{sem}}\mathcal L_{\mathrm{semantic/OOD}}\\
&+\lambda_{\mathrm{temp}}\mathcal L_{\mathrm{temporal}}.
\end{aligned}
$$

---

## 15. Occupancy 中最重要的设计：观测与补全分离

对于任意位置 $x$，建议模型显式维护三个量。

### 15.1 传感器支持的占据

$$
p_{\mathrm{obs}}(x).
$$

表示由 LiDAR return、多视角一致性或其他真实 sensor evidence 支持的 occupied probability。

### 15.2 生成先验补全的占据

$$
p_{\mathrm{prior}}(x).
$$

表示 SAM 3D-like prior 对不可见部分给出的 shape completion hypothesis。

### 15.3 补全不确定性

$$
u_{\mathrm{completion}}(x).
$$

表示该位置是多大程度上的生成猜测，以及不同合理三维解释之间的分歧。

例如：

```text
                一辆只观测到正面的汽车

     LiDAR returns                         不可见背面
         ● ● ●                                  ?
       ┌─────────┐                         ┌─────────┐
       │ 正面表面 │                         │ prior   │
       └─────────┘                         │ completion
                                          └─────────┘

     p_obs 高                              p_obs 低
     p_prior 高                            p_prior 可能高
     uncertainty 低                        uncertainty 高
```

如果需要得到一个融合概率，可以使用 evidence-dependent gate：

$$
p_{\mathrm{final}}(x)
=
1-
\left(1-p_{\mathrm{obs}}(x)\right)
\left(1-\gamma(x)p_{\mathrm{prior}}(x)\right),
$$

其中 $\gamma(x)$ 由可见性、teacher/student calibration、多视角一致性和 LiDAR conflict 共同决定。

但在模型内部和输出接口中，最好仍然保留 $p_{\mathrm{obs}}$、$p_{\mathrm{prior}}$ 和 uncertainty 三个独立 channel，而不是过早合并。

---

## 16. OOD 分数不能直接来自 SAM 3D 重建置信度

重建得好不代表是 ID，重建得差也不一定是 OOD。例如：

- 严重遮挡的普通汽车可能重建很差；
- 一个外形类似熟悉类别的 OOD 物体可能被重建得很好；
- 图像模糊、过曝会影响 reconstruction，但不改变物体类别；
- 强生成先验可能把 OOD 物体投影到最近的 familiar shape mode。

因此，entity-level OOD score 应综合：

$$
s_k^{\mathrm{OOD}}
=
f
\left(
d_{\mathrm{visual}},
d_{\mathrm{prototype}},
r_{\mathrm{multi-view}},
r_{\mathrm{LiDAR}},
v_{\mathrm{generation}},
u_{\mathrm{semantic}}
\right),
$$

其中：

| 信号 | 含义 |
|---|---|
| $d_{\mathrm{visual}}$ | entity visual feature 与已知视觉分布的距离 |
| $d_{\mathrm{prototype}}$ | 与 ID class prototypes 的距离 |
| $r_{\mathrm{multi-view}}$ | 生成形状投影到其他相机后的 mask/depth residual |
| $r_{\mathrm{LiDAR}}$ | 生成形状与 LiDAR surface/free-space 的冲突 |
| $v_{\mathrm{generation}}$ | 多次生成或 ensemble 的 shape variance |
| $u_{\mathrm{semantic}}$ | open-vocabulary semantic uncertainty |

特别值得研究的是：

> 当 SAM 3D prior 无法同时解释物体的多视角图像、LiDAR 表面和自由空间时，这种几何—视觉不一致能否作为 entity-level OOD signal？

这比单独读取 Gaussian opacity 或生成模型 confidence 更有意义。

---

## 17. LiDAR 应该提供哪些监督？

### 17.1 Surface consistency

观测到的 LiDAR points 应该落在预测 occupancy/Gaussian surface 附近：

$$
\mathcal L_{\mathrm{surface}}
=
\frac{1}{|\mathcal P|}
\sum_{p\in\mathcal P}
d\left(p,\mathcal G_{\mathrm{entity}}\right).
$$

### 17.2 Free-space constraint

在 sensor origin 与 LiDAR return 之间，射线穿过的空间应保持 free：

$$
\mathcal L_{\mathrm{free}}
=
\sum_{r\in\mathcal R}
\sum_{x\in r_{\mathrm{free}}}
p_{\mathrm{occ}}(x).
$$

这对约束 SAM 3D-like hallucination 特别重要：生成先验可以补全不可见区域，但不能占据 LiDAR 已明确穿过的自由空间。

### 17.3 Metric layout

LiDAR 可以帮助监督：

- translation $t$；
- metric scale $s$；
- orientation $R$；
- Gaussian centers 和 covariance；
- entity 3D extent。

因此我们可以让 camera 决定“这是什么、是否有物体”，LiDAR 决定“它准确在哪里、表面在哪里、哪里确定是空的”。

---

## 18. Multi-view 和 Temporal 应怎样补上？

SAM 3D 的 released pipeline 主要处理单图单物体。我们的 occupancy 模型需要将它扩展成 persistent entity。

### 18.1 Multi-view entity aggregation

同一物体可能出现在多个相机中。可以先将每个相机中的 mask feature 聚合到同一个 entity query：

$$
e_k
=
\operatorname{Aggregate}
\left(
f_{k,1}^{\mathrm{cam}},
\ldots,
f_{k,V}^{\mathrm{cam}},
g_k^{\mathrm{LiDAR}}
\right).
$$

然后只生成一套 world/ego-space Gaussian object，而不是每个相机独立生成一份物体。

### 18.2 Persistent temporal entity

对于时间 $t$ 的 entity：

$$
e_{k,t}
=
\operatorname{TemporalUpdate}
\left(
e_{k,t-1},
F_t^{\mathrm{camera}},
F_t^{\mathrm{LiDAR}}
\right).
$$

它需要维护：

- identity；
- pose；
- velocity；
- accumulated shape evidence；
- semantic/OOD belief；
- observed/completed region mask。

随着车辆运动，同一物体新的表面逐渐被观察到。模型应该使用新 evidence 修正旧的 completion，而不是永久保留第一次生成的形状。

---

## 19. 它与 S2GO、GaussianFormer3D 的关系

### 19.1 与 S2GO

S2GO 的核心是：

$$
\text{parent spatial query}
\rightarrow
\text{small child Gaussian bundle}.
$$

SAM 3D 的核心是：

$$
\text{image + object mask}
\rightarrow
\text{canonical shape + layout}
\rightarrow
\text{SLAT}
\rightarrow
\text{large Gaussian object}.
$$

二者可以结合为：

$$
\text{persistent object entity query}
\xrightarrow[\text{SAM 3D prior distillation}]{}
\text{compact semantic/OOD Gaussian bundle}.
$$

也就是说，我们保留 S2GO 的实时、紧凑和 temporal query structure，再使用 SAM 3D 教会 query 具有真正的 object-level shape prior。

### 19.2 与 GaussianFormer3D

GaussianFormer3D 更像是用大量 scene-level Gaussian primitives 覆盖整个 occupancy space；一个 Gaussian query 不一定对应一个物体。

SAM 3D 是显式 object-centric 的：一个输入 mask 对应一个完整 object asset。

所以可以让：

- GaussianFormer3D-style branch 负责 stuff/background；
- SAM 3D-inspired entity branch 负责 object/OOD；
- 两者最后共同 rasterize 成统一的 semantic occupancy grid。

---

## 20. 三种可落地的接入强度

### 方案 A：Offline Teacher Distillation（最推荐）

```text
训练时：SAM 3D teacher → shape/layout/SLAT supervision → compact student
推理时：只运行 student
```

优点：

- 推理成本最低；
- 容易和现有 occupancy backbone 对接；
- 能将 SAM 3D prior 转成适合自动驾驶的输出字段；
- 可以加入 LiDAR、multi-view、temporal 和 OOD 监督。

风险：

- teacher 的错误和熟悉化偏差可能被 student 学到；
- 需要 carefully gate teacher supervision；
- SAM 3D license 和离线预处理成本需要检查。

### 方案 B：只在线运行 SAM 3D Stage 1

```text
top-K high-risk entities
          │
          ▼
SAM 3D geometry/layout stage
          │
          ▼
coarse shape prior
```

不运行纹理和完整 Gaussian refinement，只使用 coarse shape/layout。

优点：比完整 pipeline 轻，并保留条件生成能力。

缺点：对多 object、20 Hz 推理仍可能太重，需要异步或低频运行。

### 方案 C：完整 SAM 3D 作为异步专家

仅对极少数高 OOD、高风险、严重遮挡的物体运行完整模型，用于：

- 事故分析；
- offline map update；
- difficult-case verification；
- pseudo-label generation；
- long-horizon planning 的低频补充。

它不应成为基础 occupancy 的必经路径。

---

## 21. 建议的实验路线

### Experiment 1：车载 LiDAR point map 是否能直接帮助 SAM 3D？

在 nuScenes、Waymo 或自有数据上选择有 instance mask 的物体，对比：

1. RGB + mask；
2. RGB + mask + MoGe point map；
3. RGB + mask + projected sparse LiDAR point map；
4. RGB + mask + completed LiDAR point map。

评价：

- metric translation error；
- scale error；
- observed surface Chamfer distance；
- novel-view mask/depth consistency；
- LiDAR free-space violation。

### Experiment 2：Stage 1 shape prior 是否已经足够？

比较：

- coarse $64^3$ shape；
- compressed SLAT；
- 4-Gaussian decoder；
- full 32-Gaussian decoder。

目标是判断 occupancy 是否真的需要完整的纹理 refinement。大概率 coarse shape/SLAT 已经包含我们最关心的几何信息。

### Experiment 3：将 SAM 3D shape 压缩成 compact Gaussian bundle

训练一个 decoder：

$$
O_T\text{ or }Z_T^{\mathrm{SLAT}}
\rightarrow
\{G_1,\ldots,G_J\},
\qquad J\in\{10,20,32,64\}.
$$

观察 occupancy IoU、可见/不可见区域 completion 和 runtime 之间的权衡。

### Experiment 4：SAM 3D prior 对 OOD 是帮助还是掩盖？

分别测试：

- 常见 ID objects；
- 遮挡的 ID objects；
- shape-near OOD；
- shape-far OOD；
- 拼接或非常规结构物体；
- 多次生成采样的 variance。

关注：

- OOD AUROC；
- FPR95；
- entity recall；
- known-class mIoU；
- free-space violation；
- observed/completed occupancy calibration。

### Experiment 5：Teacher distillation 与在线 teacher 对比

| 设置 | SAM 3D 何时运行 | 研究问题 |
|---|---|---|
| No SAM 3D | 从不 | 基线 |
| Offline pseudo label | 数据预处理 | shape supervision 是否有效 |
| Frozen teacher during training | 训练期 | feature/shape distillation 是否有效 |
| Stage-1 top-K online | 部分推理 | 在线生成是否值得成本 |
| Full SAM 3D online | 少量案例 | 作为上界和专家模型 |

---

## 22. 一个可以写进论文的方法表述

中文版本：

> 我们将 SAM 3D Objects 视为物体级生成式三维先验，而非在线 occupancy predictor。训练阶段，该先验指导视觉主导的 persistent entity queries 学习 canonical shape 和紧凑的 child Gaussian 表示；LiDAR 用于锚定公制 layout、观测表面与自由空间。模型显式区分传感器支持的 observed occupancy 和生成先验产生的 shape completion，并在 entity level 预测 open-set semantics、OOD probability 与补全不确定性。

英文版本：

> We use SAM 3D Objects as an object-level generative 3D prior rather than an online occupancy predictor. The prior teaches visually initiated persistent entity queries to decode canonical shapes into compact Gaussian bundles, while LiDAR anchors metric layouts, observed surfaces, and ray-based free space. Our representation explicitly separates sensor-supported occupancy from prior-based shape completion and predicts open-set semantics, OOD probability, and completion uncertainty at the entity level.

---

## 23. 最终判断

SAM 3D 与我们的研究方向确实高度相关，尤其是它证明了：

> 一个 object mask 和强视觉特征，可以被提升为具有 canonical geometry、metric layout 和 Gaussian representation 的完整三维 entity。

但是，SAM 3D 面向的是单物体的高质量生成；OOD occupancy 面向的是多物体、实时、公制、安全可校准的场景理解。二者之间必须增加：

1. camera 主导、LiDAR 提供几何的非对称融合；
2. multi-view entity aggregation；
3. persistent temporal entity tracking；
4. semantic/open-vocabulary/OOD heads；
5. compact occupancy Gaussian decoder；
6. LiDAR surface 和 ray free-space constraints；
7. observed occupancy 与 generative completion 的显式分离；
8. 对 generative familiarization bias 的检测与校准。

所以最值得做的不是复制 SAM 3D，而是把它转化为下面的核心模块：

```text
SAM 3D object-level 3D prior
              │ distillation
              ▼
persistent visual entity query
              │ queries LiDAR geometry
              ▼
compact semantic/OOD Gaussian bundle
              │
              ▼
observed occupancy + uncertain shape completion
```

这条路线可以自然连接我们已经研究的 S2GO、GaussianFormer3D、camera-LiDAR asymmetric attention 和 entity-level OOD modeling，同时也提供一个与现有 voxel-wise OOD occupancy 明显不同的研究贡献点。

---

## 24. 后续设计文档

针对“static LiDAR map + 3D bbox 标注”的实际数据形式，Scene/Entity 双分支、ownership、bbox rasterization 和推理合成的详细设计见：

- [Scene Branch + Entity Branch 设计](./SCENE_ENTITY_BRANCH_DESIGN.md)
