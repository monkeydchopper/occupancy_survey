# FlashOcc：用 2D BEV 高效预测完整 3D Occupancy

> 论文：[FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin](https://arxiv.org/abs/2311.12058)  
> 官方代码：[Yzichen/FlashOCC](https://github.com/Yzichen/FlashOCC)  
> 本文重点回答：FlashOcc 到底预测的是 2D 还是 3D？`channel-to-height` 如何实现？高架、桥洞和悬空物体能不能表示？它是否适合我们的 Camera–LiDAR、binary occupancy 系统？

---

## 0. 先给结论

FlashOcc 的核心不是一个全新的 camera backbone，而是一个非常高效的 **3D occupancy 解码方式**：

$$
\boxed{
\text{不在整个网络中维护 } H\times W\times Z\times C
\text{ 的 dense 3D feature，}
\quad
\text{只在最后把 2D BEV channel 解码成高度。}
}
$$

它的主干数据流是：

```text
多相机图像
    │
    ▼
Image Backbone + FPN
    │
    ▼
LSS / BEVPool：图像特征提升并聚合到 BEV
    │
    ▼
2D BEV Feature                 [B, C, H, W]
    │
    ▼
2D BEV Encoder
    │
    ▼
Channel-to-Height Head         [B, Z × K, H, W]
    │ reshape
    ▼
完整 3D Occupancy Logits       [B, H, W, Z, K]
```

其中：

- $H,W$ 是 BEV 平面网格；
- $Z$ 是高度层数；
- $K$ 是语义类别数，包含 free/empty 类；
- 原始 FlashOcc 是 camera-only occupancy；
- 它输出的确实是完整的 3D voxel occupancy，不是 BEV segmentation，也不是一张 height map。

对于我们的任务，它最大的吸引力是：

> 我们只预测 binary non-ground occupancy，因此 head 只需输出 $Z$ 个 logit，而不是 $Z\times K$ 个语义 logit；这比原始 FlashOcc 还要轻。

---

# Part I：FlashOcc 想解决什么问题

## 1. Dense 3D occupancy 为什么慢

传统 voxel-based occupancy 网络通常在中间层保留：

$$
F_{3D}\in\mathbb{R}^{B\times C\times Z\times H\times W}.
$$

随后用 3D convolution 或 3D transformer 处理。它的问题是：

1. **显存大**：每个 voxel 都有 $C$ 维 feature；
2. **计算大**：3D convolution 同时在 $x,y,z$ 三个方向滑动；
3. **很多空间是空的**：道路场景的 3D 网格高度维通常非常稀疏；
4. 最终虽然只需要每个 voxel 的类别概率，中间却一直维护高维 3D feature。

粗略比较中间 feature 数量：

$$
\text{Dense 3D feature: } O(HWZC),
$$

$$
\text{FlashOcc BEV feature: } O(HWC).
$$

FlashOcc 的判断是：驾驶场景大部分结构可以先在 BEV 中理解，昂贵的高度展开可以推迟到输出层。

## 2. 核心设计原则

FlashOcc 将网络分成两件事：

1. 在二维 BEV 平面中完成大部分空间上下文推理；
2. 在最后一个轻量 head 中，为每个 BEV cell 一次预测整根垂直柱。

这就是论文标题中的：

$$
\boxed{\text{Channel-to-Height, C2H}}
$$

它并没有消灭最终的 $H\times W\times Z$ occupancy 输出，而是消灭了昂贵的 **高维 dense 3D 中间特征**。

---

# Part II：从图像到 BEV

## 3. 多相机图像编码

设有 $N$ 个相机：

$$
I_n\in\mathbb{R}^{3\times H_I\times W_I},
\qquad n=1,\ldots,N.
$$

每张图像先经过共享的 backbone 和 neck：

$$
F_n^{img}=E_{img}(I_n).
$$

官方 M0 配置使用：

- 6 个 nuScenes 相机；
- 图像输入 $256\times704$；
- ResNet-50；
- FPN 输出 image feature；
- LSS view transformer；
- 2D BEV encoder；
- `BEVOCCHead2D`。

因此更准确地说：

> FlashOcc 主要创新在 occupancy 表示和 decoder，不在 image encoder 或 view transformation。官方实现建立在 BEVDet/BEVDepth/BEVDet4D 生态上。

## 4. 深度分布与 LSS lifting

单张 RGB 图像没有直接深度。LSS 类 view transformer 为每个图像 feature pixel 预测离散深度分布：

$$
P_n(d\mid u,v),
$$

同时预测 context feature：

$$
F_n^{ctx}(u,v).
$$

两者相乘得到视锥中的 feature：

$$
F_n^{frustum}(u,v,d)
=
P_n(d\mid u,v)F_n^{ctx}(u,v).
$$

利用相机内外参，把 $(u,v,d)$ 变换到 ego/world 坐标，再通过 BEV pooling 聚合：

$$
F_{BEV}(x,y)
=
\operatorname{Pool}
\left(
\left\{F_n^{frustum}(u,v,d)\right\}_{n,u,v,d}
\right).
$$

官方 M0 的 view-transformer 配置中 `collapse_z=True`，因此这一阶段输出：

$$
F_{BEV}\in\mathbb{R}^{B\times C\times H\times W},
$$

而不是 dense 3D feature volume。

### 一个容易混淆的地方

这里有两个不同的“高度离散化”：

1. LSS 聚合阶段把垂直方向压到二维 BEV；
2. occupancy head 最终重新输出 $Z$ 个明确的 occupancy 高度层。

所以：

> `collapse_z=True` 不意味着输出没有高度，只表示中间的 BEV feature 不保留显式的 $z$ 轴。

## 5. 2D BEV encoder

聚合后的 BEV feature 用普通 2D CNN 处理：

$$
\widetilde F_{BEV}=E_{BEV}^{2D}(F_{BEV}).
$$

二维 BEV encoder 可以有效建模：

- 道路的空间连续性；
- 车辆之间的平面关系；
- 可行驶区域与路侧结构；
- 远近区域上下文；
- 多相机在 BEV 中的统一表示。

主要计算都发生在二维，因此 FlashOcc 能比 dense 3D decoder 快很多。

---

# Part III：Channel-to-Height 到底如何工作

## 6. 它不是单纯的 `reshape`

假设 BEV encoder 的输出为：

$$
\widetilde F_{BEV}\in\mathbb{R}^{B\times C\times H\times W}.
$$

官方 `BEVOCCHead2D` 首先做一个 $3\times3$ 二维卷积：

$$
U=\operatorname{Conv2D}_{3\times3}(\widetilde F_{BEV}),
$$

然后对每个 BEV cell 的 feature vector $U(x,y)$ 使用一个小 MLP：

$$
h(x,y)=W_2\,\operatorname{Softplus}(W_1U(x,y)+b_1)+b_2.
$$

MLP 最后一层的输出维度是：

$$
Z\times K.
$$

因此：

$$
h(x,y)\in\mathbb{R}^{ZK}
\quad\Longrightarrow\quad
L(x,y,z,k)\in\mathbb{R}^{Z\times K}.
$$

最后只是把预测好的 $ZK$ 个数重新解释成高度和类别两个维度：

```text
[B, C, H, W]
      │ 3×3 Conv2D
      ▼
[B, C', H, W]
      │ per-cell MLP: C' → 2C' → Z×K
      ▼
[B, H, W, Z×K]
      │ reshape
      ▼
[B, H, W, Z, K]
```

所以 C2H 的本质是：

$$
\boxed{
\text{用一个 BEV column feature，预测该 }(x,y)
\text{ 位置所有高度和类别的 logits。}
}
$$

## 7. 为什么每个高度能学出不同结果

对固定的 $(x,y)$，不同高度对应 MLP 最后一层的不同输出神经元：

$$
\ell_{z,k}(x,y)=w_{z,k}^{\top}h'(x,y)+b_{z,k}.
$$

因此 $z=0$、$z=1$、$z=2$ 的预测参数并不相同。训练标签会告诉这些输出位置：

- 哪些高度是 free；
- 哪些高度是 road；
- 哪些高度是 car；
- 哪些高度是 manmade structure。

固定 ego 坐标系中，每个输出下标都有稳定的物理高度，所以网络可以学习“某种 BEV feature 对应怎样的垂直占用剖面”。

## 8. 一个具体例子

假设一个 BEV cell 有 8 个高度层，预测结果可以是：

```text
z7  free
z6  bridge / manmade      █
z5  bridge / manmade      █
z4  free
z3  free
z2  car                   █
z1  car                   █
z0  road                  █
```

注意 $z=3,4$ 可以是空的，而更高的 $z=5,6$ 又被占据。

这说明它不是：

- 从地面向上填满的柱子；
- 每个 $(x,y)$ 只预测一个最高高度；
- 把二维 semantic map 直接 extrusion；
- 只能表示落地物体。

它能够输出非连续的垂直 occupancy pattern。

---

# Part IV：它能不能表示高架和 overhead structure

## 9. 表示能力：可以

FlashOcc 的最终输出是：

$$
O\in\{0,\ldots,K-1\}^{H\times W\times Z}.
$$

同一个 $(x,y)$ 下的多个高度格子可以独立选择 occupied/free 或不同语义类。因此在表示层面，它能够表示：

- 桥面下方为空、上方被占据；
- 隧道顶棚；
- 悬空交通灯；
- 树冠与树干；
- 高处招牌；
- 卡车货箱下方局部空隙；
- 多个不相连的垂直表面。

所以“FlashOcc 不能表示 3D occupancy”是不准确的。

## 10. 推理能力：有明显上限

虽然输出格式能表示上述结构，但 FlashOcc 的中间 feature 是：

$$
F(x,y)\in\mathbb{R}^{C},
$$

而不是：

$$
F(x,y,z)\in\mathbb{R}^{C}.
$$

这意味着同一垂直柱中所有高度的预测共享同一个 BEV feature。原始 2D head 中也没有显式的：

- $z$ 方向 convolution；
- 相邻高度 voxel attention；
- 三维局部几何传播；
- 不同高度层之间的独立高维 feature。

因此它更容易在以下情况遇到问题：

1. 多层道路、复杂立交和桥下车辆；
2. 上下两层结构在 BEV 中严重重叠；
3. 很薄的横杆、线缆、交通牌；
4. 深度歧义严重的远处目标；
5. 垂直几何远比 nuScenes 常规道路场景复杂的区域。

准确的评价应当是：

> FlashOcc 的输出是完整 3D，但它用纵向建模能力换取了很高的速度和很低的中间显存。

## 11. 地面移除对 overhead 的影响

如果我们的 non-ground occupancy GT 是从 LiDAR map 中移除地面得到的，需要区分：

- ego 所在道路表面；
- 桥上另一层道路或桥体；
- 隧道顶棚；
- 其他水平但并非当前地面的表面。

如果 preprocessing 把“所有近似水平面”都删掉，桥面和 overhead structure 就会被误删。此时任何模型都学不出来。

更合理的是把 ground 定义为：

$$
\text{与 ego 当前可行驶表面连接的局部地面},
$$

而不是：

$$
\text{所有水平结构}.
$$

---

# Part V：训练目标和推理输出

## 12. 原始 FlashOcc 的监督

官方 M0 head 预测：

$$
L\in\mathbb{R}^{B\times H\times W\times Z\times K}.
$$

然后对每个有效 voxel 使用 semantic cross-entropy：

$$
\mathcal L_{occ}
=
-\frac{1}{|\mathcal V|}
\sum_{v\in\mathcal V}
\log p(y_v\mid v).
$$

$\mathcal V$ 可以由 camera-visible mask 限定。官方 `BEVOCCHead2D` 中：

1. 把 $[B,H,W,Z,K]$ 展平；
2. 用 `mask_camera` 过滤或加权；
3. 计算 cross-entropy；
4. 推理时对类别维 softmax，再 argmax。

## 13. 默认 M0 的张量尺寸

官方 `flashocc-r50-M0.py` 的关键配置是：

| 项目 | 配置 |
|---|---:|
| Camera 数量 | 6 |
| 图像输入 | $256\times704$ |
| BEV 范围 | $x,y\in[-40,40]$ m |
| BEV 分辨率 | $0.4$ m |
| BEV 网格 | $200\times200$ |
| 输出高度层 | $Z=16$ |
| View-transformer 输出 channel | 64 |
| Head 输入/内部 channel | 128 |
| Occupancy 类别数 | 18 |

因此最终 logits 大致为：

$$
[B,200,200,16,18].
$$

需要再次强调：LSS 配置中的单个聚合高度 bin 与最终的 16 个 occupancy 高度层不是同一回事。后者由 `Dz=16` 和 occupancy GT 定义。

## 14. 原始 FlashOcc 是否需要 LiDAR input

默认 M0 是 **camera-only inference**：

```python
input_modality = dict(
    use_lidar=False,
    use_camera=True,
)
```

官方数据 pipeline 会加载点云并生成 `gt_depth`，但基础 `BEVDetOCC + LSSViewTransformer` 的在线 occupancy feature 提取仍然来自图像。不同配置需要分开看：

- 基础 FlashOcc M0：推理依赖 camera；
- BEVDepth/BEVStereo 类变体：训练时可用 LiDAR 生成 depth supervision；
- 这种 LiDAR depth supervision 不等于推理时把 LiDAR 作为第二输入模态；
- FlashOcc 的 C2H head 本身与输入模态无关，也可接在 Camera–LiDAR fused BEV 后面。

---

# Part VI：FlashOcc 为什么快

## 15. 它省掉的是什么

FlashOcc 仍然必须产生最终 dense occupancy logits：

$$
H\times W\times Z\times K.
$$

它省掉的是每个 voxel 上的大 feature vector：

```text
Dense 3D decoder:
    [B, C, Z, H, W] ── many 3D layers ──> [B, K, Z, H, W]

FlashOcc:
    [B, C, H, W]    ── many 2D layers ──> [B, Z×K, H, W]
```

当 $C$ 很大且 3D decoder 有多层时，差距非常明显。

粗略地，普通 convolution 的计算趋势是：

$$
\text{2D Conv}\sim HW\,C_{in}C_{out}k^2,
$$

$$
\text{3D Conv}\sim HWZ\,C_{in}C_{out}k^3.
$$

实际速度还受 backbone、view transformation、BEV pooling、TensorRT 和内存访问影响，但 C2H 确实避免了最贵的一段 dense 3D processing。

## 16. 官方速度数字应该怎样理解

官方 README 报告的代表性结果为：

| 方法 | Backbone / 输入 | mIoU | FPS | FLOPs | 参数量 |
|---|---|---:|---:|---:|---:|
| BEVDetOCC 1-frame | R50, $256\times704$ | 31.60 | 92.1 | 241.76 G | 29.02 M |
| FlashOcc M0 1-frame | R50, $256\times704$ | 31.95 | 197.6 | 154.1 G | 39.94 M |

但这个数字必须带上测试条件：

$$
\boxed{\text{RTX 3090 + TensorRT + FP16}}
$$

因此不能把 `197.6 FPS` 直接与以下结果横向比较：

- PyTorch eager FP32；
- A100 上的论文速度；
- 包含数据预处理或不包含预处理的速度；
- 使用更高图像分辨率或 temporal frames 的模型；
- Camera–LiDAR fusion 模型。

它能说明的是：

> C2H 这一 occupancy decoder 的部署效率非常高；不代表任何硬件和实现上都必然达到 197.6 FPS。

对我们要求的 $>20$ FPS，FlashOcc-style binary head 是现实且有余量的选择，但整体速度最终更可能由 image backbone、view transform 和 LiDAR encoder 决定。

---

# Part VII：与几类方法的区别

## 17. FlashOcc 不是 Gaussian occupancy

| 方法 | 中间场景表示 | 3D 输出如何产生 | 主要优点 | 主要代价 |
|---|---|---|---|---|
| Dense voxel network | dense 3D feature | 3D conv/transformer | 强三维局部建模 | 慢、显存大 |
| FlashOcc | dense 2D BEV feature | channel-to-height | 极高效率、部署友好 | 高度信息被压缩 |
| GaussianFormer3D | sparse 3D Gaussian queries | Gaussian-to-voxel splatting | 稀疏、显式 3D primitive | query 更新和 splatting 有成本 |
| S2GO | temporal Gaussian queries | query propagation + Gaussian rendering/splatting | temporal streaming | 系统复杂，速度口径不同 |
| FlashOcc-style fusion | Camera–LiDAR 2D BEV | channel-to-height | 适合实时多模态 | 融合质量决定上限 |

FlashOcc 没有显式 entity、object query 或 Gaussian。它是规则网格上的 dense prediction，只是把高维 3D computation 换成 2D computation。

## 18. 它和 height map 也不同

Height map 通常是：

$$
\hat z(x,y)\in\mathbb R,
$$

每个 BEV cell 只给一个高度，不能描述一根柱子里多个分离结构。

FlashOcc 是：

$$
\hat o(x,y,z),\qquad z=1,\ldots,Z.
$$

因此它能表达完整垂直 occupancy profile。Ground height head 和 FlashOcc occupancy head 是互补的两个任务，不应混为一谈。

---

# Part VIII：怎样用于我们的模型

## 19. 我们当前目标

我们的设计已经收敛为三个 head：

1. **Ground Height Head**：预测当前地表高度；
2. **Binary Non-ground Occupancy Head**：预测所有被占据的非地面空间，不赋语义；
3. **Known-class BEV Detection Head**：只识别训练时定义的 known objects。

unknown 不作为一个 semantic class 学习，而是由：

$$
\text{occupied but not explained by known detection}
$$

得到。

## 20. Camera-led Camera–LiDAR backbone

建议首先把两个模态统一到二维 BEV：

```text
Multi-view RGB                         LiDAR
      │                                  │
      ▼                                  ▼
Image Backbone + LSS              Lightweight Voxel/Pillar Encoder
      │                                  │
      ▼                                  ▼
Camera BEV F_cam                    LiDAR BEV F_lidar
      │                                  │
      └──────── Camera-led gated fusion ─┘
                         │
                         ▼
                    F_fused [B,C,H,W]
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   Ground Height    Binary C2H Occ    Known CenterHead
```

一个轻量、非对称的融合形式是：

$$
F_{fused}
=
F_{cam}
+
\sigma(g(F_{cam}))\odot P(F_{lidar}),
$$

其中：

- $F_{cam}$ 是主干语义表示；
- $P$ 对齐 LiDAR feature 的 channel；
- gate 由 camera feature 决定；
- LiDAR 提供几何、距离和可见表面证据；
- 不需要对 $H\times W$ 所有 cell 做全局 cross-attention。

为了保留一部分垂直几何，LiDAR BEV encoder 可以在压缩高度前保留少量 height bins，再沿 channel 拼入 BEV，而不必构造大型 dense 3D tensor。

## 21. Binary FlashOcc head

原始 semantic FlashOcc 的输出 channel 是：

$$
Z\times K.
$$

我们的 occupancy 不带属性，所以可以改为：

$$
Z
$$

个 binary logit：

```text
F_fused [B,C,H,W]
       │
       ▼
Depthwise/standard 3×3 Conv2D
       │
       ▼
1×1 Conv2D: C → Z
       │
       ▼
L_occ [B,Z,H,W]
       │ permute
       ▼
L_occ [B,H,W,Z]
```

数学形式：

$$
p_{occ}(x,y,z)
=
\sigma\left(\ell_{occ}(x,y,z)\right).
$$

损失可以先用最简单稳健的组合：

$$
\mathcal L_{bin-occ}
=
\lambda_{bce}\mathcal L_{BCE}
+
\lambda_{lovasz}\mathcal L_{Lovasz}.
$$

如果 free voxel 数量远大于 occupied voxel，需要：

- positive weighting 或 focal loss；
- 只在定义明确的 valid/observed mask 内监督；
- 评估 precision、recall、IoU 和距离分段 recall，而不只看整体 accuracy。

### 不需要 unknown class

训练标签只需要：

$$
y_{occ}(x,y,z)\in\{0,1\}.
$$

无论是 known vehicle、带框的 unknown、无框 sign、cone、植被还是其他障碍，只要 GT 中为 occupied，都进入同一个 positive 集合。

这与“unknown 没有统一外观特征”的目标是一致的：网络学的是几何占用，不是一个虚构的 unknown semantic prototype。

## 22. Ground height conditioning

Ground head 输出：

$$
\hat z_g(x,y),
$$

还可以输出 uncertainty：

$$
\hat u_g(x,y).
$$

occupancy head 可以使用相对地面高度：

$$
z_{rel}=z-\hat z_g(x,y),
$$

帮助模型区分：

- 当前道路；
- 高于地面的物体；
- 地下或无效 voxel；
- 坡道上的一致物理高度。

早期实现可对 $\hat z_g$ 使用 `stop_gradient` 再喂给 occupancy head，避免 occupancy loss 通过捷径扰乱 ground head。

## 23. Known detection 与 unknown occupancy

Known-object head 只预测已知类的 BEV boxes：

$$
\mathcal B_{known}=\{(B_i,c_i,s_i)\}.
$$

把高置信 known box rasterize 成 voxel mask：

$$
M_{known}(v)
=
\max_i
s_i\mathbf 1[v\in B_i].
$$

推理时可以定义 residual unknown score：

$$
p_{unk}(v)
=
p_{occ}(v)\left(1-M_{known}(v)\right).
$$

这不是把 unknown 训练成一个类别，而是做集合差：

$$
\boxed{
\text{Unknown Occupancy}
=
\text{All Occupancy}
-
\text{Known-object Explanation}
}
$$

为减少 detector 漏检造成的假 unknown，实际部署时建议输出三个图层而不是立刻硬分类：

- `occupancy_probability`；
- `known_explanation_probability`；
- `residual_unknown_probability`。

## 24. 对 overhead structure 的轻量增强

如果验证集显示 bridge、tunnel、overhead sign 明显较差，不必立刻换成完整 3D backbone。可以只在低维 occupancy logits 上加入一个很小的 3D refinement：

$$
L_0\in\mathbb{R}^{B\times1\times Z\times H\times W},
$$

$$
L
=
L_0
+
\operatorname{Conv3D}_{3\times3\times3}(L_0).
$$

这里 3D conv 只处理 1 到 4 个 channel 的 logits，不处理 128/256-channel dense voxel feature，因此成本小得多。

更便宜的版本是在每个 $(x,y)$ 上做纵向 1D convolution：

$$
L(x,y,:)
\leftarrow
L(x,y,:)
+
\operatorname{Conv1D}_z(L(x,y,:)).
$$

建议按以下顺序实验：

1. 纯 binary C2H；
2. C2H + vertical 1D refinement；
3. C2H + tiny logit-space 3D refinement；
4. 只有前三者确实不足时，再考虑 dense 3D feature decoder。

---

# Part IX：工程建议与实验设计

## 25. 一个适合 20 FPS 目标的起始配置

下面是起始点，不是硬性结论：

| 模块 | 建议 |
|---|---|
| Camera | 6 views, $256\times704$ 起步 |
| Image backbone | ResNet-18 / EfficientNet-lite / 已有轻量 BEV encoder |
| BEV 网格 | 先用 $0.4$ m 分辨率 |
| BEV channel | 96 或 128 |
| LiDAR branch | PointPillars 或 SECOND-lite |
| Fusion | concat + Conv2D baseline；再试 camera-gated residual |
| Occupancy | binary C2H，$Z=16$ 或按真实高度范围设置 |
| Detection | shared BEV 上的 CenterHead |
| Ground | height + uncertainty head |
| Precision | FP16；部署后再测 INT8 |

速度必须在目标硬件上拆分测量：

```text
image preprocess
+ image backbone
+ view transform / BEV pooling
+ LiDAR voxelization and encoder
+ BEV fusion
+ three heads
+ post-processing
= end-to-end latency
```

不要只记录 occupancy head 的 latency。

## 26. 最关键的消融实验

### A. C2H 是否足够

| 实验 | Head |
|---|---|
| A0 | Binary C2H |
| A1 | C2H + vertical Conv1D |
| A2 | C2H + tiny logit 3D refinement |
| A3 | 小型 dense 3D decoder 上界 |

单独统计：

- 普通道路；
- bridge/overpass；
- tunnel；
- overhead sign；
- 远距离；
- 被遮挡区域。

### B. LiDAR 是否真正发挥几何作用

| 实验 | Fusion |
|---|---|
| B0 | Camera-only |
| B1 | Camera BEV + LiDAR BEV concat |
| B2 | Camera-gated LiDAR residual |
| B3 | B2 + LiDAR height-bin feature |

除了整体 IoU，还应看 depth/距离分段结果和细小障碍 recall。

### C. Ground 分支是否有帮助

| 实验 | Ground 使用方式 |
|---|---|
| C0 | 不使用 ground head |
| C1 | 多任务监督但不条件化 occupancy |
| C2 | relative-height conditioning + stop-gradient |
| C3 | 联合梯度 conditioning |

## 27. 失败模式检查清单

训练出结果后，优先检查：

- 是否把低置信 free space 当成 occupied；
- 是否只会从地面向上“挤满”物体；
- 桥下的 free gap 是否保留；
- GT ground removal 是否误删桥面/顶棚；
- known detector 的漏检是否被误报成 unknown；
- LiDAR 稀疏区域是否过度依赖单点；
- camera 不可见但 map GT 有标注的区域是否被当作负样本；
- temporal map 与当前传感器时刻是否对齐；
- 速度测试是否包含 voxelization、BEVPool 和 post-processing。

---

# Part X：最终判断

## 28. FlashOcc 对我们是否合适

我的判断是：**非常合适作为第一版高效 occupancy decoder，但不应原封不动照搬整个 camera-only 系统。**

应该保留：

1. 大部分推理发生在 2D BEV；
2. 最后使用 channel-to-height 输出完整 3D occupancy；
3. 避免大型 dense 3D feature volume；
4. 部署友好的 convolutional head。

应该修改：

1. Camera-only 改成 camera-led Camera–LiDAR BEV fusion；
2. semantic $Z\times K$ 输出改成 binary $Z$ 输出；
3. 增加 ground height head；
4. 增加 known-class CenterHead；
5. 用 residual explanation 得到 unknown occupancy；
6. 只在证明确有需要时加入 tiny vertical/3D refinement。

最终推荐结构是：

$$
\boxed{
\begin{aligned}
F_{cam},F_{lidar}
&\rightarrow F_{fused}^{2D}\\
F_{fused}^{2D}
&\rightarrow \text{Ground Height}\\
F_{fused}^{2D}
&\rightarrow \text{Binary Channel-to-Height Occupancy}\\
F_{fused}^{2D}
&\rightarrow \text{Known-class BEV Detection}\\
\text{Unknown}
&=\text{Occupancy}-\text{Known Explanation}.
\end{aligned}
}
$$

一句话总结：

> **FlashOcc 不是把 3D occupancy 降级成 2D，而是把昂贵的三维特征推理压缩到二维 BEV，再用一个很轻的高度解码器恢复完整 3D 输出。它可以表示 overhead structure；真正需要验证的是，共享的 column feature 是否足以理解我们数据中的复杂垂直结构。**

---

# Part XI：Occupancy 下游怎样使用

## 29. Occupancy 的最终价值是什么

Occupancy 最核心的下游用途不是生成一张好看的三维图，而是给规划系统提供一个统一的、包含未知障碍的碰撞空间。

我们当前三个 head 的输出可以这样使用：

```text
Ground Height       Binary 3D Occupancy       Known Detection
     │                       │                       │
     └──────────────┬────────┘                       │
                    ▼                                │
        Ego-relative obstacle voxels                 │
                    │                                │
                    ├──────── Known box 解释 ────────┘
                    │
                    ▼
       Known obstacles + Residual unknown obstacles
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
       碰撞检测   可行驶空间   规划代价图
```

其中：

- Detection 告诉规划器“这是什么、怎样运动”；
- Occupancy 告诉规划器“这里有没有东西、车辆能不能通过”；
- Ground height 提供局部地面参考，避免把高处结构错误投影成地面障碍。

## 30. 直接进行三维碰撞检测

Occupancy head 输出：

$$
p_{occ}(x,y,z)\in[0,1].
$$

对于一条候选轨迹 $\tau$，把车辆在未来每个时刻的车身包络联合起来，得到 swept volume：

$$
\mathcal V(\tau)
=
\bigcup_t
\operatorname{VehicleVolume}(x_t,y_t,z_t,\theta_t).
$$

轨迹风险可以定义为 swept volume 内的最大 occupancy probability：

$$
R_{max}(\tau)
=
\max_{v\in\mathcal V(\tau)}p_{occ}(v).
$$

也可以累积风险：

$$
R_{sum}(\tau)
=
\sum_{v\in\mathcal V(\tau)}
w(v,\tau)p_{occ}(v).
$$

如果：

$$
R_{max}(\tau)>\tau_{hard},
$$

就把这条轨迹判为不可行。

这样，即使障碍没有类别或没有 detection bbox，也能参与碰撞判断，例如：

- 未见过的施工设备；
- 掉落物；
- 无框交通牌或路侧结构；
- 不常见车辆；
- detector 漏检的 known object；
- 无法被单个 bbox 准确描述的墙面和异形物体。

## 31. 为什么不能把所有高度直接做 `max`

最简单的 BEV 压缩是：

$$
O_{BEV}(x,y)
=
\max_z p_{occ}(x,y,z).
$$

但这个结果不能直接用于车辆碰撞，因为它会把高架、交通灯和高处标牌全部当成地面障碍。

正确做法是结合 ground height：

$$
z_{rel}
=
z-\hat z_g(x,y).
$$

只在车辆真实占用的相对高度范围中查询 occupancy：

$$
p_{collision}(x,y)
=
\max_{
z_{rel}\in[z_{ego}^{min},z_{ego}^{max}]
}
p_{occ}(x,y,z).
$$

```text
高处广告牌      occupied，但高于车辆碰撞范围
══════════════════════════════════════

车辆高度上限    ──────────────────────
                 规划碰撞检查区域
当前地面        ──────────────────────
```

还可以根据不同任务使用不同高度区间：

| 下游任务 | 查询的相对高度 |
|---|---|
| 车轮/底盘碰撞 | 接近地面的低高度区域 |
| 完整车身碰撞 | 从底盘到车顶 |
| 车顶净空判断 | 接近车顶及其安全余量 |
| 地图可视化 | 全部高度 |

因此 ground-height head 不只是一个训练辅助任务，它是 occupancy 转化为规划约束的重要坐标参考。

## 32. 从三维 occupancy 生成规划 cost map

规划器通常不会直接消费完整的 $H\times W\times Z$ tensor，而是使用从中生成的多个 BEV cost layers。

### 32.1 硬碰撞层

$$
C_{hard}(x,y)
=
\mathbf 1
\left[
p_{collision}(x,y)>\tau_h
\right].
$$

它表示车辆绝对不能进入的区域。

### 32.2 软代价层

为了让轨迹和障碍保持安全距离，可以先根据车辆 footprint 和安全余量对 obstacle mask 做膨胀，再计算距离变换：

$$
C_{soft}(x,y)
=
\exp
\left(
-\frac{d(x,y,\mathcal O)^2}{2\sigma^2}
\right).
$$

```text
障碍中心        规划代价
   ███            1.0
  ▓███▓           0.8
 ▒▓███▓▒          0.5
░▒▓███▓▒░         0.2
```

一条轨迹的整体代价可以写成：

$$
J(\tau)
=
J_{route}(\tau)
+
\lambda_{occ}
\sum_t C_{soft}(x_t,y_t)
+
\lambda_{collision}
\mathbf 1[\tau\text{ 与 occupancy 碰撞}].
$$

### 32.3 不同高度的风险可以不同

低矮路沿、车辆高度障碍和 overhead structure 的风险不同，可以定义高度相关权重：

$$
C(x,y)
=
\max_z
w\bigl(z-\hat z_g(x,y)\bigr)
p_{occ}(x,y,z).
$$

这比无条件沿高度取最大值更合理。

## 33. Occupancy 与 known detection 的结合

Known detection 提供：

- 物体类别；
- 三维 bbox；
- 朝向；
- 速度；
- tracking ID。

Occupancy 提供：

- 更细的 occupied geometry；
- 不属于 known 类的障碍；
- 任意形状的路侧结构；
- detection 漏检后的安全兜底。

将高置信 known boxes rasterize 成 voxel explanation mask：

$$
M_{known}(v)
=
\max_i
s_i\mathbf 1[v\in B_i].
$$

Residual unknown occupancy 定义为：

$$
p_{unknown}(v)
=
p_{occ}(v)
\left(1-M_{known}(v)\right).
$$

```text
Occupied voxel
      │
      ├── 被高置信 known box 解释
      │       └── known car / pedestrian / ...
      │
      └── 无法被 known box 解释
              └── residual unknown obstacle
```

这里的 unknown 不是 occupancy head 学到的 semantic class，而是：

$$
\boxed{
\text{有东西占据空间，但 known detector 无法可靠解释。}
}
$$

### 不要过早做硬集合差

如果 known detector 的 confidence 较低，直接将 box 内 occupancy 全部删除会掩盖 detector 错误。更稳健的是保留三个概率层：

1. `occupancy_probability`；
2. `known_explanation_probability`；
3. `residual_unknown_probability`。

规划器仍然以总 occupancy 作为安全依据，unknown residual 主要用于风险升级、日志记录和后续 object discovery。

## 34. 从 residual occupancy 生成临时 unknown objects

主网络不需要 entity head。下游可以按需要对 residual occupancy 做聚类：

```text
Residual unknown voxels
          │
          ▼
3D connected components / DBSCAN
          │
          ▼
Unknown obstacle components
          │
          ├── centroid
          ├── occupied voxel shape
          ├── oriented/axis-aligned 3D bbox
          ├── nearest distance
          ├── estimated size
          └── confidence
```

例如，下游可以生成：

```text
Unknown obstacle #7
center:       (18.2 m, -3.1 m)
size:         (1.4 m, 0.8 m, 1.1 m)
nearest dist: 17.5 m
confidence:   0.82
class:        unknown
```

这里的 bbox 是 occupancy 的后处理结果，不要求 occupancy 网络预先把空间划分为 entities。

需要注意，相邻但属于不同物体的 voxel 可能粘连；一个稀疏物体也可能被分裂成多个 component。因此这些临时 instances 适合用于规划和 tracking，不应被当作高质量 instance segmentation GT。

## 35. Temporal tracking 与动态 occupancy

单帧 occupancy 回答的是：

> 当前什么位置被占据？

动态规划还需要回答：

> 这些占用接下来会移动到哪里？

Known objects 可以直接使用 detector/tracker 输出的速度。Residual unknown 可以先采用轻量后处理：

1. 用 ego pose 将连续帧 occupancy 对齐到同一坐标系；
2. 对 unknown connected components 做时间关联；
3. 估计 centroid velocity；
4. 将 component 向未来时刻传播。

简单的速度估计为：

$$
\hat{\mathbf v}_j
=
\frac{
\mathbf c_j^t-\mathbf c_j^{t-1}
}{\Delta t}.
$$

未来占用可以近似为：

$$
O_j^{t+\Delta t}
=
\operatorname{Warp}
\left(
O_j^t,
\hat{\mathbf v}_j\Delta t
\right).
$$

第一版系统没有必要立即训练复杂的 future occupancy 网络。先用 detection tracking 加 unknown component tracking，就可以验证 occupancy 对规划是否真正有增益。

## 36. 必须区分两种 unknown

这是安全使用 occupancy 时最重要的概念之一。

### 36.1 Unknown object

传感器已经看到该空间被占据，但它不能被 known detector 解释：

$$
\text{occupied but unexplained}.
$$

### 36.2 Unobserved space

由于相机视野、遮挡、距离或 LiDAR 稀疏性，系统没有充分观测该空间：

$$
\text{insufficiently observed}.
$$

这两者不能合并。尤其不能把“没有预测出 occupied”自动解释为“确定 free”。

即使我们不训练 unknown semantic class，下游仍然最好保留一个独立的 observability/confidence：

$$
q_{obs}(x,y,z)\in[0,1].
$$

| Occupancy | Observability | 下游解释 |
|---:|---:|---|
| 高 | 高 | 确认障碍 |
| 低 | 高 | 确认 free |
| 高 | 低 | 可能存在障碍，保守处理 |
| 低 | 低 | 未充分观测，不能直接当作 free |

`observability` 不一定需要额外的神经网络 head，可以根据以下信息构造：

- camera frustum；
- image depth confidence；
- LiDAR ray traversal；
- LiDAR return coverage；
- 距离；
- 遮挡关系；
- temporal observation history。

它不是物体的 semantic attribute，而是说明当前传感器是否有能力可靠判断该 voxel。

## 37. 推荐的系统输出接口

感知模型最好不要只输出硬二值 occupancy，而是保留概率和辅助信息：

```python
PerceptionOutput = {
    "occupancy_probability":  Tensor[H, W, Z],
    "ground_height":          Tensor[H, W],
    "ground_uncertainty":     Tensor[H, W],
    "observability":          Tensor[H, W, Z],
    "known_objects":          List[Box3D],
}
```

下游的 occupancy adapter 再生成规划需要的格式：

```python
PlanningInput = {
    "collision_cost_bev":     Tensor[H, W],
    "hard_obstacle_mask":     Tensor[H, W],
    "known_tracks":           List[Track],
    "unknown_components":     List[UnknownObstacle],
    "unobserved_mask":        Tensor[H, W],
}
```

建议把这个 adapter 做成独立模块。这样可以在不重新训练网络的情况下调整：

- occupancy threshold；
- 车辆碰撞高度范围；
- safety dilation；
- known box explanation threshold；
- unknown component 最小尺寸；
- unobserved region 的保守程度。

## 38. 对我们当前项目最合适的第一版下游

第一版不必做复杂 prediction，可以采用：

```text
1. 输出 binary 3D occupancy probability
2. 用 ground height 转换成 relative-height occupancy
3. 按 ego body height 生成 collision BEV
4. 按车辆 footprint + safety margin 做膨胀
5. known boxes 提供类别、速度和 tracking
6. occupancy 减去 soft known explanation，得到 residual unknown
7. 对 residual unknown 做 connected components
8. 将总 occupancy cost、known tracks 和 unknown components 交给 planner
```

这里有一个重要原则：

$$
\boxed{
\text{规划安全约束使用 total occupancy，}
\quad
\text{unknown residual 用于解释、跟踪和风险升级。}
}
$$

不能因为某块 occupancy 被 known box 解释，就从碰撞图里删除它；known/unknown 的划分只改变属性解释，不应该破坏总障碍几何。

最终，OOD occupancy 对下游最重要的问题不是：

> 这个物体究竟属于哪个 unknown 类？

而是：

$$
\boxed{
\text{这个 unexplained occupied region
是否侵入车辆未来的 swept volume？}
}
$$

这也是 binary occupancy 加 known detection 设计的直接工程价值。

---

## 参考资料

1. [FlashOcc 论文（arXiv）](https://arxiv.org/abs/2311.12058)
2. [FlashOcc 官方代码](https://github.com/Yzichen/FlashOCC)
3. [官方 `BEVOCCHead2D` 实现](https://github.com/Yzichen/FlashOCC/blob/main/projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py)
4. [官方 M0 配置](https://github.com/Yzichen/FlashOCC/blob/main/projects/configs/flashocc/flashocc-r50-M0.py)
