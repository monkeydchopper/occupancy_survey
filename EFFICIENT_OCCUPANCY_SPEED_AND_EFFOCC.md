# 高效 Occupancy 方法速度比较与 EFFOcc 融合方式详解

> 本文解决两个问题：  
> 1. EFFOcc 到底怎样融合 Camera 和 LiDAR？  
> 2. S2GO 是否是目前最快的 occupancy 方法，我们的 20 FPS 系统应该选择哪条技术路线？

相关资料：

- [EFFOcc 论文](https://arxiv.org/abs/2406.07042)
- [EFFOcc 官方代码](https://github.com/synsin0/effocc)
- [FlashOcc 论文](https://arxiv.org/abs/2311.12058)
- [FlashOcc 官方代码](https://github.com/Yzichen/FlashOCC)
- [UltimateDO 论文](https://arxiv.org/abs/2409.11160)
- [Panoptic-FlashOcc 论文](https://arxiv.org/abs/2406.10527)
- [S2GO 论文](https://arxiv.org/abs/2506.05473)

---

## 0. 结论先行

### EFFOcc 的融合

EFFOcc 不是 attention fusion。它有两次 Camera–LiDAR 信息交互：

1. 将投影 LiDAR depth 输入 Camera LSS view transformer，帮助图像特征预测深度并生成 Camera BEV；
2. 将 Camera BEV 与 LiDAR BEV 沿 channel 拼接，再使用一个 `3×3 Conv2D + BN + ReLU` 融合。

核心公式：

$$
F_{\mathrm{cam-BEV}}
=
\operatorname{LSS}
\left(
F_{\mathrm{img}},D_{\mathrm{LiDAR}}
\right),
$$

$$
F_{\mathrm{fused}}
=
\operatorname{Conv}_{3\times3}
\left(
[F_{\mathrm{cam-BEV}};F_{\mathrm{LiDAR-BEV}}]
\right).
$$

### 速度比较

S2GO 不是所有 occupancy 方法中最快的。更准确的说法是：

> **S2GO 是 Gaussian/query/streaming occupancy 中非常快的一档；纯 2D BEV + channel-to-height 的 FlashOcc 类方法可以明显更快。**

### 对我们当前任务的建议

我们当前只需要：

1. Ground Height Head；
2. Binary Non-ground Occupancy Head；
3. Known-object BEV Detection Head。

最适合的路线是：

$$
\boxed{
\text{Light Camera--LiDAR BEV Fusion}
+
\text{Binary Channel-to-Height}
+
\text{CenterHead}
}
$$

不需要 S2GO 的 temporal query、Gaussian decoder 和 Gaussian-to-voxel splatting。

---

# Part I：EFFOcc 融合方式详解

## 1. EFFOcc 的整体数据流

```text
Multi-view Images                         LiDAR
        │                                   │
        ▼                                   ├──────────────┐
ResNet-18 + FPN                            │              │
        │                                  │              │
        ▼                                  ▼              ▼
Image Features                    Projected Depth    Voxelization
        │                                  │              │
        └──────── concat ──────────────────┘              ▼
                     │                            SparseEncoder
                     ▼                                  │
            LSS Depth Network                         ▼
                     │                         LiDAR BEV Feature
                     ▼                                  │
            BEVPoolV2 / Lift-Splat                     │
                     │                                  │
                     ▼                                  │
           Camera BEV Feature                           │
                     │                                  │
                     └──────── channel concat ──────────┘
                                      │
                                      ▼
                            3×3 Conv2D + BN + ReLU
                                      │
                                      ▼
                            Fused 2D BEV Feature
                                      │
                                      ▼
                             SECOND + SECOND FPN
                                      │
                                      ▼
                           Channel-to-Height Occ Head
```

EFFOcc 的主要设计目标是用 2D BEV operators 替代 dense 3D feature processing：

- Camera 中间表示被压缩到 BEV；
- LiDAR sparse voxel feature 被压缩到 BEV；
- 跨模态融合在 BEV 上完成；
- 融合后使用 2D SECOND/SECOND-FPN；
- 最后才将 channel 解码为 height dimension。

---

## 2. Camera 分支

### 2.1 图像特征

官方 OpenOccupancy R18 配置使用：

- 六路相机；
- 输入分辨率 $256\times704$；
- ResNet-18；
- Custom FPN；
- 图像 feature 输出 128 channels；
- LSS 下采样率为 8。

图像数据流：

$$
I
\rightarrow
\operatorname{ResNet18}
\rightarrow
\operatorname{FPN}
\rightarrow
F_{\mathrm{img}}.
$$

官方配置：

- [`effocc-fusion-r18.py`](https://github.com/synsin0/effocc/blob/main/configs/effocc_openoccupancy/effocc-fusion-r18.py)

### 2.2 LiDAR depth 注入图像分支

EFFOcc 将 LiDAR points 投影到每个相机，得到 sparse depth map：

$$
D_{\mathrm{LiDAR}}(u,v).
$$

LiDAR depth 经过一个小型 CNN：

$$
F_{\mathrm{depth}}
=
E_{\mathrm{depth}}
\left(D_{\mathrm{LiDAR}}\right).
$$

官方代码中的 channel 变化大致为：

```text
LiDAR depth: 1 channel
        │
        ▼
1×1 Conv: 1 → 8
        │
        ▼
5×5 Conv: 8 → 32
        │
        ▼
5×5 Conv: 32 → 64
```

然后与 image feature 拼接：

$$
F_{\mathrm{img-depth}}
=
[F_{\mathrm{img}};F_{\mathrm{depth}}].
$$

Depth Network 输出：

1. 每个像素的 depth-bin distribution；
2. 用于 lifting 的 image context feature。

$$
P(d\mid u,v),
\qquad
F_{\mathrm{context}}(u,v).
$$

对应源码：

- [`view_transformer.py`](https://github.com/synsin0/effocc/blob/main/mmdet3d/models/necks/view_transformer.py)

### 2.3 LSS/BEVPoolV2

利用标定参数和预测 depth distribution，将多相机 feature 投影到 BEV：

$$
F_{\mathrm{cam-BEV}}
=
\operatorname{BEVPoolV2}
\left(
P(d\mid u,v),F_{\mathrm{context}}
\right).
$$

配置中 `collapse_z=True`，所以输出不是 dense 3D camera volume，而是 2D BEV feature：

$$
F_{\mathrm{cam-BEV}}
\in
\mathbb R^{B\times64\times H\times W}.
$$

这可以称为：

$$
\boxed{
\text{LiDAR-depth-guided Camera BEV lifting}
}
$$

它并不是只把 image feature 附着到 sparse LiDAR returns；模型仍然预测完整 depth distribution，再通过 LSS 聚合到 BEV。

---

## 3. LiDAR 分支

LiDAR 路径为：

```text
LiDAR points
      │
      ▼
Hard Voxelization
      │
      ▼
HardSimpleVFE / Mean Feature Encoding
      │
      ▼
SparseEncoder
      │ collapse height
      ▼
LiDAR BEV Feature
```

可以写成：

$$
F_{\mathrm{LiDAR-BEV}}
=
E_{\mathrm{LiDAR}}(P).
$$

官方 R18 OpenOccupancy 配置使用：

- `HardSimpleVFE`；
- `SparseEncoder`；
- sparse spatial shape 约为 $41\times4096\times4096$；
- encoder 输出最终被压缩为 BEV feature。

在 fuser 输入处：

$$
F_{\mathrm{LiDAR-BEV}}
\in
\mathbb R^{B\times256\times H\times W}.
$$

---

## 4. 真正的 BEV 融合层

Camera BEV 和 LiDAR BEV 已经对齐到同一 spatial grid：

$$
F_{\mathrm{cam-BEV}}(:,x,y)
\quad\text{和}\quad
F_{\mathrm{LiDAR-BEV}}(:,x,y)
$$

描述同一块世界空间。

融合操作是 channel concatenation：

$$
F_{\mathrm{concat}}
=
[F_{\mathrm{cam-BEV}};F_{\mathrm{LiDAR-BEV}}].
$$

根据官方配置：

$$
64+256=320\text{ channels}.
$$

然后使用：

$$
F_{\mathrm{fused}}
=
\operatorname{ReLU}
\left(
\operatorname{BN}
\left(
\operatorname{Conv}_{3\times3}
(F_{\mathrm{concat}})
\right)
\right).
$$

输出：

$$
F_{\mathrm{fused}}
\in
\mathbb R^{B\times256\times H\times W}.
$$

源码核心等价于：

```python
x = torch.cat([img_bev_feats, pts_bev_feats], dim=1)
x = conv2d_3x3(x)
x = batch_norm(x)
x = relu(x)
```

对应：

- [`convfuse2d.py`](https://github.com/synsin0/effocc/blob/main/mmdet3d/models/fuser/convfuse2d.py)

因此 EFFOcc 没有：

- cross-attention；
- deformable attention；
- Camera query；
- LiDAR key/value；
- token matching；
- iterative fusion；
- modality-specific gate。

---

## 5. 融合后的 BEV Encoder

融合 feature 继续进入检测网络中常见的：

```text
Fused BEV
    │
    ▼
SECOND Backbone
    │
    ▼
SECOND FPN
    │
    ▼
Occupancy BEV Feature
```

$$
F_{\mathrm{occ-BEV}}
=
\operatorname{SECOND-FPN}
\left(
\operatorname{SECOND}
(F_{\mathrm{fused}})
\right).
$$

官方 OpenOccupancy R18 配置中的 occupancy head 接收约 512-channel BEV feature。

这也说明 EFFOcc 更接近 LiDAR-centric design：

- LiDAR 分支较重；
- Camera 分支较轻；
- LiDAR depth 参与 Camera lifting；
- BEV 融合后继续使用 SECOND/SECOND-FPN；
- Camera BEV 64 channels，LiDAR BEV 256 channels。

因此 EFFOcc 不是视觉主导的非对称融合，而更像：

> **以 LiDAR BEV backbone 为主体，加入一个 LiDAR-depth-guided 的轻量 Camera branch。**

---

## 6. Channel-to-Height Occupancy Head

EFFOcc 保持中间 feature 为 2D BEV：

$$
F_{\mathrm{occ-BEV}}
\in
\mathbb R^{B\times C\times H\times W}.
$$

Occupancy head 先使用 2D convolution，再对每个 BEV cell 输出：

$$
Z\times C_{\mathrm{semantic}}
$$

个 logits：

$$
F_{\mathrm{occ-BEV}}(x,y)
\rightarrow
O(x,y)
\in
\mathbb R^{ZC_{\mathrm{semantic}}}.
$$

最后 reshape：

$$
O
\in
\mathbb R^{B\times H\times W\times Z\times C_{\mathrm{semantic}}}.
$$

官方 OpenOccupancy 配置使用：

- $H=W=512$；
- $Z=40$；
- semantic classes $=17$。

对应 head：

- [`bev_occ_head.py`](https://github.com/synsin0/effocc/blob/main/mmdet3d/models/dense_heads/bev_occ_head.py)

---

## 7. EFFOcc 的融合属于什么类型？

EFFOcc 同时包含：

### Image-space early fusion

$$
F_{\mathrm{img}}+D_{\mathrm{LiDAR}}
\rightarrow
F_{\mathrm{cam-BEV}}.
$$

LiDAR depth 在 image view transformer 中帮助预测深度。

### BEV-space middle fusion

$$
F_{\mathrm{cam-BEV}}+F_{\mathrm{LiDAR-BEV}}
\rightarrow
F_{\mathrm{fused}}.
$$

主融合发生在 feature-aligned BEV space。

因此最准确的描述是：

> **LiDAR-depth-guided view transformation + BEV feature concat-convolution。**

---

## 8. 为什么简单 concat 仍然有效？

跨模态对应关系已经通过标定和 view transformation 显式处理：

$$
(u,v,d)
\xrightarrow[	ext{calibration}]{}
(x,y)_{\mathrm{BEV}}.
$$

因此融合时不需要再在全局范围内搜索 correspondence。同一个 BEV cell 中已经包含：

- Camera semantic/appearance feature；
- LiDAR geometry feature。

`3×3 Conv2D` 还能同时查看周围 cell，吸收局部标定误差和空间上下文。

这种融合的优势：

- 简单；
- 部署友好；
- 容易 TensorRT 优化；
- 不构建 attention matrix；
- 不维护 dense 3D feature volume。

缺点：

- 没有显式 camera/LiDAR reliability modeling；
- 两个模态发生冲突时只依靠 convolution 自己学习；
- 没有规定 camera 或 LiDAR 谁主导；
- 局部标定偏差和稀疏 LiDAR 可能导致 feature mismatch；
- 视觉覆盖不完整时可能发生 fusion degradation。

EFFOcc 论文也报告，在部分 Waymo 设置上，LiDAR-only 甚至略优于 fusion model，并将其归因于视觉覆盖不完整和模态冲突等因素。

---

## 9. EFFOcc 为什么没有达到 20 FPS？

EFFOcc 的 fuser 本身非常轻，但整个系统仍然包含：

- 六相机 backbone；
- LiDAR-depth-guided LSS；
- BEVPoolV2；
- 高分辨率 LiDAR voxelization；
- SparseEncoder；
- SECOND；
- SECOND-FPN；
- 高分辨率 multi-class occupancy output。

论文报告：

| 方法 | 参数量 | FPS |
|---|---:|---:|
| RadOcc-LC | 135.39M | 0.3 |
| EFFOcc-R18 | 21.35M | 5.6 |
| EFFOcc-Swin-B | 111.48M | 1.8 |

所以 EFFOcc 的“efficient”是相对于重型 3D fusion baseline：

$$
0.3\ \mathrm{FPS}
\rightarrow
5.6\ \mathrm{FPS}.
$$

这实现了数量级加速，但原始配置仍未达到 20 FPS。

主要原因不是 `ConvFuser2D`，而是 encoder、view transformation、LiDAR voxel resolution 和高分辨率 semantic output。

---

# Part II：不同 Occupancy 方法的速度比较

## 10. 为什么不能只看一个 FPS 数字？

不同论文中的 FPS 可能使用不同：

- GPU：A100、RTX 4090、RTX 3090；
- backend：PyTorch、TensorRT；
- precision：FP32、FP16、INT8；
- input resolution；
- camera 数量；
- camera-only 或 Camera–LiDAR；
- single-frame 或 temporal；
- occupancy range 和 voxel resolution；
- semantic classes 数量；
- camera visibility mask；
- 是否包含数据预处理和后处理。

因此下面的表用于理解数量级和设计趋势，不能作为严格的公平排名。

---

## 11. 公开速度数据

| 方法 | 核心表示 | 硬件/后端 | 报告 FPS |
|---|---|---|---:|
| FlashOcc M0 | 2D BEV → channel-to-height | RTX 3090, TensorRT FP16 | 197.6 |
| FlashOcc M1 | 2D BEV → channel-to-height | RTX 3090, TensorRT FP16 | 152.7 |
| Panoptic-FlashOcc-Depth-tiny | 2D BEV | A100, PyTorch FP32 | 43.9 |
| Panoptic-FlashOcc-Depth-tiny-Pano | 2D BEV + panoptic | A100, PyTorch FP32 | 39.8 |
| S2GO-Small | 900 queries, 9000 Gaussians | RTX 4090 | 26.1 |
| S2GO-Base | 1800 queries, 36000 Gaussians | RTX 4090 | 19.6 |
| S2GO-Small | 900 queries, 9000 Gaussians | A100 | 20.8 |
| S2GO-Base | 1800 queries, 36000 Gaussians | A100 | 14.5 |
| SparseOcc | sparse occupancy | A100 | 12.5 |
| EFFOcc-R18 Fusion | Camera–LiDAR 2D BEV fusion | 论文测速环境 | 5.6 |
| GaussianWorld | temporal Gaussian | RTX 4090 | 4.4 |
| BEVFormer | dense/query BEV | RTX 4090 comparison | 3.3 |
| SurroundOcc | dense occupancy | RTX 4090 comparison | 3.3 |
| TPVFormer | tri-plane transformer | RTX 4090 comparison | 2.9 |
| GaussianFormer-2 | Gaussian queries | RTX 4090 comparison | 2.8 |
| GaussianFormer | Gaussian queries | RTX 4090 comparison | 2.7 |

注意：FlashOcc 官方明确说明其 197.6 FPS 是 TensorRT FP16；S2GO 的主要表格属于 PyTorch research model benchmark。两者不能直接当作同一部署条件下的公平对比。

---

## 12. S2GO 是不是最快？

答案分两层。

### 12.1 在 Gaussian/query/streaming 类方法中

是最快的一档。

在 S2GO 对 SurroundOcc-nuScenes 方法统一进行的 RTX 4090 测试中：

| 方法 | FPS |
|---|---:|
| GaussianFormer | 2.7 |
| GaussianFormer-2 | 2.8 |
| GaussianWorld | 4.4 |
| S2GO-Base | 19.6 |
| S2GO-Small | 26.1 |

因此：

$$
\frac{26.1}{4.4}
\approx
5.9.
$$

S2GO 相对 GaussianWorld 约 5.9 倍加速。

### 12.2 在所有 occupancy 方法中

不是。

FlashOcc/Panoptic-FlashOcc 这种 2D BEV + channel-to-height 路线可以达到更高 FPS。

因此应表述为：

$$
\boxed{
\text{S2GO is among the fastest Gaussian/query-based occupancy models.}
}
$$

而不是：

$$
\text{S2GO is the fastest occupancy model.}
$$

---

## 13. S2GO 的 latency breakdown

S2GO-Small 使用：

- 900 queries；
- 每个 query 10 个 Gaussians；
- 总计 9000 Gaussians。

论文在 A100 上报告的主要 latency：

| 模块 | 延迟 |
|---|---:|
| Image backbone | 11.54 ms |
| Temporal Transformer | 22.79 ms |
| Gaussian prediction | 2.22 ms |
| Query propagation | 1.45 ms |

合计：

$$
11.54+22.79+2.22+1.45
=
38.00\ \mathrm{ms}.
$$

还需考虑 Gaussian-to-voxel splatting 和系统开销，因此论文在 Occ3D/A100 上报告约 20.8 FPS。

最大的瓶颈是：

$$
\boxed{
\text{Temporal Transformer: }22.79\ \mathrm{ms}
}
$$

而不是 Gaussian prediction 本身。

所以“稀疏 query/Gaussian 表示”并不等于整个 pipeline 一定比简单 2D CNN 更快。

---

## 14. FlashOcc 为什么快？

FlashOcc 始终维护：

$$
F_{\mathrm{BEV}}
\in
\mathbb R^{B\times C\times H\times W}.
$$

最后通过 2D convolution 输出 height channels：

$$
O
=
\operatorname{Conv2D}
(F_{\mathrm{BEV}})
\in
\mathbb R^{B\times (ZC_{\mathrm{sem}})\times H\times W}.
$$

再 reshape：

$$
O
\rightarrow
\mathbb R^{B\times H\times W\times Z\times C_{\mathrm{sem}}}.
$$

它没有：

- temporal transformer；
- query interaction；
- Gaussian decoding；
- Gaussian-to-voxel splatting；
- dense 3D convolution。

这是它获得极高部署速度的根本原因。

---

## 15. 为什么我们的 Binary Occupancy 可以更轻？

Semantic FlashOcc 需要输出：

$$
Z\times C_{\mathrm{semantic}}
$$

channels。

我们的 Occupancy Head 只输出：

$$
Z
$$

个 binary logits。

例如：

$$
Z=20,
\qquad
C_{\mathrm{semantic}}=18.
$$

Semantic output channels：

$$
20\times18=360.
$$

Binary output channels：

$$
20.
$$

因此最后一层输出和内存写入可以显著减少。

---

## 16. 方法速度的粗略层级

在实现和输入条件接近时，可以粗略理解为：

```text
最快
  │
  │ 2D BEV + Channel-to-Height
  │ FlashOcc / UltimateDO-style
  │
  │ Sparse Query + Gaussian
  │ S2GO
  │
  │ Sparse Voxel / BEV Transformer
  │ SparseOcc 等
  │
  │ Dense Gaussian / Dense 3D CNN
  │ GaussianFormer / heavy fusion occupancy
  ▼
最慢
```

这只是计算结构上的趋势，不代表 accuracy、temporal capability 或 output quality 的完整排序。

---

# Part III：我们应该怎样设计 20 FPS 系统？

## 17. 我们当前任务不需要 S2GO 的复杂模块

当前任务已经简化为：

```text
Head 1: Ground Height
Head 2: Binary Non-ground Occupancy
Head 3: Known-object BEV Detection
```

不需要：

- semantic occupancy head；
- unknown class head；
- entity/scene decomposition；
- temporal Gaussian queries；
- Gaussian shape decoder；
- Gaussian-to-voxel splatting。

因此，为了速度不应优先选择 S2GO。

---

## 18. 推荐实时架构

```text
Multi-view Camera                         LiDAR
        │                                   │
        ▼                                   ▼
Light Visual Backbone              PointPillars / SECOND-Lite
        │                                   │
        ▼                                   ▼
Camera BEV Feature                   LiDAR BEV Feature
        │                                   │
        └──── Camera-led Gated Fusion ◄──────┘
                          │
                          ▼
                Shared 2D BEV Feature
                  [B, C, H, W]
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      Ground Height    Binary Occ    Known Detection
          Head           Head         CenterHead
            │             │             │
            ▼             ▼             ▼
       z_ground       [B,Z,H,W]      known bboxes
```

### Ground Head

$$
(\hat z_g,\hat u_g)
=
\operatorname{Conv2D}(F_{\mathrm{BEV}}).
$$

### Binary Occupancy Head

$$
O
=
\operatorname{Conv}_{C\rightarrow Z}
(F_{\mathrm{BEV}}).
$$

$$
p_{\mathrm{occ}}
=
\sigma(O).
$$

### Known Detection Head

使用 CenterPoint/CenterHead 风格输出：

- known-class heatmap；
- center offset；
- bbox size；
- height；
- yaw。

不使用 DETR query decoder。

---

## 19. 我们是否应该照搬 EFFOcc 融合？

可以保留其基本思想，但不建议原样复制完整配置。

### 可以保留

1. LiDAR depth 帮助 Camera view transformation；
2. Camera/LiDAR 在 BEV feature level 融合；
3. 全部使用 2D operators；
4. channel-to-height output；
5. detection 与 occupancy 共享 BEV encoder。

### 不建议照搬

1. 0.025 m 的高分辨率 LiDAR voxelization；
2. 大 spatial shape SparseEncoder；
3. 完整 SECOND + SECOND-FPN；
4. $512\times512\times40\times17$ semantic output；
5. 没有 reliability gate 的直接 concat fusion。

---

## 20. 更适合我们的 Camera-led 轻量融合

EFFOcc 使用：

$$
F_{\mathrm{fused}}
=
\operatorname{Conv}
([F_{\mathrm{cam}};F_{\mathrm{LiDAR}}]).
$$

如果希望 Camera 是主路径，可以改成：

$$
G_{\mathrm{cam}}
=
\sigma
\left(
\operatorname{Conv}_{1\times1}(F_{\mathrm{cam}})
\right),
$$

$$
F_{\mathrm{fused}}
=
F_{\mathrm{cam}}
+
G_{\mathrm{cam}}
\odot
P(F_{\mathrm{LiDAR}}),
$$

其中：

$$
P(F_{\mathrm{LiDAR}})
=
\operatorname{Conv}_{1\times1}(F_{\mathrm{LiDAR}}).
$$

特点：

- Camera feature 是 residual 主路径；
- LiDAR 提供 geometry correction；
- Camera gate 决定注入多少 LiDAR 信息；
- 没有 attention matrix；
- 全部使用部署友好的 Conv2D；
- 计算量与 concat-convolution 接近。

也可以只在 Known Detection Head 的少量 candidate centers 上增加局部 cross-attention，而不是在整个 occupancy grid 上运行 attention。

---

## 21. 推荐第一版配置

以约 $102.4\text{ m}\times102.4\text{ m}$ 感知范围为例：

| 模块 | 推荐配置 |
|---|---|
| Camera | 6 路，$256\times704$ |
| Camera backbone | ResNet-18 或同级轻量网络 |
| LiDAR encoder | PointPillars 或 SECOND-Lite |
| BEV grid | $256\times256$，约 0.4 m |
| BEV channels | 96 或 128 |
| Height bins | 16～24 |
| BEV trunk | 2D depthwise-separable residual blocks |
| Ground Head | DWConv + $1\times1$ Conv |
| Occupancy Head | DWConv + Conv $C\rightarrow Z$ |
| Detection Head | CenterHead |
| Temporal | 第一版不用 |
| Precision | FP16 |
| Batch | 1 |

如果最终评测要求 $512\times512\times40$，可以考虑：

- coarse global grid + near-field refinement；
- 在 0.4 m BEV 上预测，再做轻量 upsampling；
- 仅对 occupied/uncertain region 做稀疏细化；
- 避免在全分辨率上反复运行宽通道 Conv2D。

---

## 22. 20 FPS 延迟预算

要达到端到端 20 FPS：

$$
T_{\mathrm{frame}}<50\ \mathrm{ms}.
$$

模型本身最好控制在：

$$
T_{\mathrm{model}}<35\text{--}40\ \mathrm{ms},
$$

给 sensor preprocessing、memory copy 和 post-processing 留出空间。

一个工程目标预算：

| 模块 | 目标延迟 |
|---|---:|
| 六相机 backbone | 12～18 ms |
| Camera-to-BEV | 4～7 ms |
| LiDAR encoder | 5～8 ms |
| Fusion + BEV trunk | 5～7 ms |
| 三个 heads | 2～4 ms |
| Box decoding/association | 1～3 ms |
| 总计 | 29～47 ms |

这些是设计预算，不是未经实测的速度承诺。最终速度必须在目标硬件上以 batch size 1、完整 sensor preprocessing 和 CUDA synchronization 实测。

---

## 23. 大视觉模型与实时性的兼容方式

SAM、DINO 等大视觉模型不适合在六路相机上直接在线运行并同时保证 20 FPS。

推荐：

```text
训练阶段
Large Visual Teacher
        │
        ▼
Feature/Objectness Distillation
        │
        ▼
Light Camera Student

推理阶段
只保留 Light Student
```

大模型用于：

- 离线 pseudo labels；
- feature distillation；
- known detector pretraining；
- hard-example mining。

在线模型只使用轻量 camera backbone。

---

## 24. 最终判断

### 关于 EFFOcc

EFFOcc 的核心贡献不是复杂融合，而是：

> 将 Camera 和 LiDAR 都压缩到 2D BEV，使用最简单的 concat-convolution 融合，再以 2D SECOND 和 channel-to-height head 完成 occupancy。

它的融合形式为：

$$
\boxed{
F_{\mathrm{fused}}
=
\operatorname{Conv}_{3\times3}
([F_{\mathrm{cam-BEV}};F_{\mathrm{LiDAR-BEV}}])
}
$$

并且 Camera BEV 在生成时已经使用 LiDAR depth 引导。

### 关于 S2GO

S2GO 是 Gaussian/query occupancy 中非常快的一档：

$$
20.8\ \mathrm{FPS}\ \text{on A100},
$$

但不是所有 occupancy 中最快。它仍然承担 temporal transformer、Gaussian decoding 和 splatting 的成本。

### 关于我们的 20 FPS 系统

对于：

- Ground Height；
- Binary Non-ground Occupancy；
- Known-object Detection；

最合理的选择是：

$$
\boxed{
\text{EFFOcc-style BEV alignment}
+
\text{FlashOcc-style binary channel-to-height}
+
\text{CenterHead known detection}
}
$$

其中只借鉴 EFFOcc 的 BEV fusion 原则，不照搬它的高分辨率 LiDAR encoder 和 semantic occupancy配置。

最终 unknown occupancy 仍然由 known detection residual 得到：

$$
p_{\mathrm{unknown-occ}}(x)
=
p_{\mathrm{occ}}(x)
\left(1-M_{\mathrm{known}}(x)\right).
$$

因此我们不需要为 unknown 增加任何额外 head，也不需要用 S2GO 的 Gaussian representation 才能实现 open-set occupancy。

