# 最终设计：EFFOcc-style Camera–LiDAR OOD Occupancy

> 状态：当前讨论收敛版本  
> 日期：2026-08-31  
> 目标：使用多相机和当前帧 LiDAR，以高于 20 FPS 为工程目标，预测可被规划直接使用的开放世界障碍空间。  
> 核心原则：**不把 unknown 训练成一个类别；先预测所有 non-ground occupancy，再由 known detection 对其中一部分进行解释。**

---

## 0. 最终结论

最终模型保留三个学习输出：

1. **Ground Height Head**：预测当前 ego 所在、与当前道路连通的支撑地面高度；
2. **Binary Non-ground Occupancy Head**：预测每个 3D voxel 是否被非地面实体占据；
3. **Known-class BEV Detection Head**：只预测定义好的 known object 类别、bbox、朝向和速度。

另有一个不需要神经网络学习的辅助量：

4. **Geometric Observability Mask**：由当前 LiDAR ray tracing，以及可选的 camera frustum/depth z-buffer 计算；主要用于训练 loss mask，也可以供下游速度控制和调试使用。

完整系统为：

```text
Multi-view RGB                              Current LiDAR
      │                                          │
      ▼                                          ├── Projected sparse depth
Image backbone + neck                            │
      │                                          ▼
      ├── Image context ──────────────── Camera depth/lifting
      │                                          │
      ▼                                          ▼
Camera BEV feature F_cam                 LiDAR geometry BEV F_lidar
      │                                          │
      └──────── concat in aligned BEV ───────────┘
                             │
                             ▼
                   3×3 Conv2D + BN + ReLU
                             │
                             ▼
                    Shared 2D BEV Encoder
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
  Ground Height Head   Binary C2H Head   Known Detection Head
       [H,W]             [H,W,Z]          boxes/classes/velocity
          │                  │                  │
          └──────────┬───────┘                  │
                     ▼                          │
      Vehicle-height clipping                   │
                     │                          │
                     ▼                          │
          2D Collision Occupancy                │
                     │                          │
                     ├── Total collision map ───┘
                     │
                     └── Occupancy - known explanation
                                  ↓
                         Residual unknown obstacles
```

最终下游使用的不是完整 3D tensor，而是针对车辆尺寸生成的二维碰撞图：

$$
p_{block}(x,y)
=
\max_{
z\in[
\hat z_g(x,y)+h_{min},
\hat z_g(x,y)+h_{max}
]
}
p_{ng}(x,y,z).
$$

但是模型仍然保留 3D occupancy，因为只有先保留高度，才能正确排除桥梁、交通灯和高处标牌等不会与车身发生碰撞的结构。

---

# Part I：已经冻结的设计决策

## 1. 核心决策表

| 问题 | 最终选择 | 原因 |
|---|---|---|
| Occupancy 是否预测 semantic class | 否 | unknown objects 没有统一语义原型，规划也不需要类别 |
| 是否训练 `unknown` 类 | 否 | unknown 是“不能被 known detector 解释”，不是稳定类别 |
| 是否使用 entity/query/instance mask | 第一版不使用 | 当前 GT 主要是 static map + bbox fill，不需要复杂 entity decomposition |
| Occupancy head 输出 | 每个 voxel 一个 binary non-ground occupancy logit | 最统一、最轻量 |
| 是否单独输出 free channel | 否 | 对 binary head，$p_{free}=1-p_{occ}$，两个 channel 冗余 |
| ignore/unobserved 是否是输出类别 | 否 | 它是训练 mask 或可见性信息，不是世界语义状态 |
| Ground 如何表达 | 独立 2D height head | 与现有 ground-removed GT 匹配，效率高，方便下游 clipping |
| 跨车型坐标统一 | 各传感器先使用原外参统一到 IMU/ego frame，再按车型固定平移到 ground-centered ego frame | 保留 IMU 轴方向，只把公共原点移到名义地面，使车辆附近 ground 约为 $z=0$ |
| Detection 预测什么 | 只预测 known classes | 类别、bbox、yaw、velocity 由 detection 提供 |
| Labeled unknown bbox 如何使用 | 进入 occupancy GT；不进入 known detection positive | 与 latent unknown 在 occupancy 中统一 |
| 中间表示 | 2D BEV | 避免 dense 3D feature，满足实时性目标 |
| 3D 输出方式 | FlashOcc-style binary Channel-to-Height | 只在最后输出 $Z$ 个高度 logit |
| Camera–LiDAR 融合 | EFFOcc-style 两阶段融合：LiDAR depth guidance + BEV concat/Conv2D | 有直接 occupancy 先例，简单、稳定且易于满足实时性 |
| Observability | 几何计算，非 semantic class | 训练时避免把未观测区域错误标成 free |
| 下游格式 | vehicle-specific 2D collision BEV | 规划器通常消费二维 cost/collision map |

## 2. 明确不做什么

第一版不做：

- $K+1$ semantic occupancy；
- 一个统一的 `unknown` semantic logit；
- entity query 或 MaskFormer-style 3D instance segmentation；
- Gaussian query、Gaussian splatting 或 temporal Gaussian propagation；
- 大型 dense 3D convolutional decoder；
- 把所有未标注、未观测位置当成 free；
- 用 known bbox 从总碰撞 occupancy 中删除障碍；
- 为了输出 bbox 再从 occupancy head 中增加 object proposal 机制。

这些模块以后可以作为研究扩展，但不属于第一版最小可行系统。

---

# Part II：任务的精确定义

## 3. 模型到底预测什么

### 3.1 Ground height

$$
\hat z_g(x,y)
\in\mathbb R.
$$

这里的 ground 不是“所有水平表面”，而是：

> 当前 ego 所在、沿合理坡度和高度变化与当前道路连通的支撑表面。

因此：

- 当前行驶道路属于 ground；
- 当前道路的坡道属于 ground；
- 与 ego 当前道路不连通的上层桥面，不应从 non-ground occupancy 中删除；
- 隧道顶棚不是 ground；
- 路侧水平平台不自动视为 ground。

这个定义对于 overhead clearance 非常重要。

### 3.2 Binary non-ground occupancy

$$
y_{ng}(x,y,z)
\in\{0,1\}.
$$

它只回答：

$$
\boxed{
\text{该 voxel 是否被非当前地面的实体或保守碰撞体积占据？}
}
$$

所有以下内容都使用相同的 positive label：

- known vehicle；
- known pedestrian；
- labeled unknown bbox；
- 未提供 bbox 的 sign、cone 或其他静态障碍；
- 墙、护栏、树木和建筑结构；
- bridge/overhead structure；
- 任何不能被 known detector 分类但确实占据空间的物体。

它不需要知道这些实体叫什么。

### 3.3 Known-object detection

$$
\mathcal B_{known}
=
\{(B_i,c_i,s_i,\mathbf v_i)\}_{i=1}^{N}.
$$

Detection head 负责：

- known class；
- 3D bbox；
- center；
- size；
- yaw；
- velocity；
- 后续 tracking ID。

### 3.4 Observability

$$
M_{obs}(x,y,z)
\in\{0,1\}
$$

或：

$$
q_{obs}(x,y,z)
\in[0,1].
$$

它表示当前传感器是否对 voxel 提供了充分证据，不表示 voxel 是否被占据。

## 4. 两种 unknown 必须区分

### Unknown object

$$
\text{occupied but unexplained by known detection}.
$$

这是我们的 OOD 目标。

### Unobserved space

$$
\text{current sensors do not provide sufficient evidence}.
$$

这是观测条件，不是新物体类别。

模型不训练 unknown object 类，也不训练 unobserved semantic 类。

---

# Part III：网络架构

## 5. 总体张量流

### 5.1 输入坐标归一化：固定的 ground-centered ego frame

不同车型的 LiDAR 和 Camera 安装高度可能不同。如果直接在各自的传感器坐标系训练，同一种物体和地面会落入不同的高度 voxel。模型在学习 occupancy 之前，还要额外适应传感器安装差异。

原系统已经把 Camera 和 LiDAR 统一到 IMU/ego coordinate。新的定义不是重做一套独立标定，而是在这个公共 IMU frame 下面再增加一个**固定的 ground-centered ego frame**：保留 IMU 坐标轴方向，只把原点移动到 IMU 下方、车辆附近的名义支撑地面。

```text
Camera ── T_I←C ──┐
                  ├── IMU/ego frame I ── fixed T_G←I ── Ground-centered frame G
LiDAR  ── T_I←L ──┘
```

因此每种车辆使用一组**固定标定变换**，把所有传感器和标签转换到统一的 canonical ego frame。该坐标系约定为：

- 坐标轴方向、handedness 与原 IMU/ego frame 保持一致；
- $x/y$ 原点位于 IMU 原点在名义地面上的垂直投影位置，或项目统一定义的 vehicle base point；
- $z=0$ 位于车辆附近的名义支撑地面；
- $+z$ 指向车辆上方；
- 对同一车辆，变换不随帧、道路坡度或车辆动态姿态改变。

设 $I$ 是原来的公共 IMU/ego frame，$G$ 是新的 ground-centered frame。对于车辆 $v$，IMU 原点的名义离地高度为 $h_{I,v}$。若只改变原点高度而不改变轴方向，则：

$$
\mathbf p_G
=
T_{G\leftarrow I_v}\mathbf p_{I_v},
\qquad
x_G=x_I,
\quad
y_G=y_I,
\quad
z_G=z_I+h_{I,v}.
$$

这里的 $h_{I,v}$ 是固定传感器/车体外参，不是世界海拔，也不是逐帧估计的 ground height。Camera 和 LiDAR 到 $G$ 的外参由原有 IMU 外参直接组合：

$$
T_{G\leftarrow L_v}
=
T_{G\leftarrow I_v}T_{I_v\leftarrow L_v},
$$

$$
T_{G\leftarrow C_{i,v}}
=
T_{G\leftarrow I_v}T_{I_v\leftarrow C_{i,v}}.
$$

因此 LiDAR normalization 和 Camera lifting 不是两套独立处理，而是共享同一个 $T_{G\leftarrow I_v}$。Camera image 和 intrinsics 不变；pixel feature 经深度 lift 到 Camera 3D frame 后，使用组合后的 $T_{G\leftarrow C_{i,v}}$ 进入 canonical space：

$$
\mathbf p_{C_i}(u,v,d)
=
dK_i^{-1}
\begin{bmatrix}
u & v & 1
\end{bmatrix}^{T},
$$

$$
\mathbf p_G
=
T_{G\leftarrow C_{i,v}}\mathbf p_{C_i}.
$$

随后根据 $(x_G,y_G,z_G)$ 执行 BEV pooling。被平移的是 feature 对应的 3D 几何坐标，不是 image feature 数值，也不能给 Camera optical depth 加 $h_{I,v}$。

如果 LiDAR 轴已经与 IMU/ego 轴对齐，也可以把它理解为先转换到原 IMU frame，再对 $z_I$ 加固定高度：

$$
z_G=z_I+h_{I,v}.
$$

这样处理后，合理的数据分布应当是：

```text
车辆附近地面                  z ≈ 0
地面上车辆、行人和其他结构      z > 0
下坡、坑洼或标定残差            允许 z < 0
远处上坡                       允许 z > 0 并随距离变化
```

这里的目的只是消除**固定安装高度和固定安装角度**差异，不是把每一帧地面拉平。明确禁止：

- 根据车辆当前 pitch/roll 每帧重新定义 ground-flat frame；
- 用单个平面强行拉平整段道路；
- 把真实坡道、路拱和远处地形变化压到 $z=0$；
- 将归一化后的负高度 clamp 到 0。

所有几何量必须使用同一个 canonical frame：

```text
LiDAR points
Camera extrinsics / Camera lifting geometry
projected LiDAR sparse depth
occupancy GT
ground-height GT
known/unknown 3D bbox
static map crop
voxel grid and output coordinates
```

Camera 外参需要由各自传感器坐标一致地组合到 $G$。不能只执行 `lidar[:, 2] += h_v`，却继续使用以原 IMU 原点为目标 frame 的 Camera lifting 外参或未变换的 GT。若现有 Camera/LiDAR 外参本来已经直接指向 ground-centered `base_link`，则不得重复增加高度。

对于 temporal fusion 或 mapping，如果定位系统输出的是 IMU global pose，则 ground-centered frame 的 global pose 为：

$$
T_{W\leftarrow G_t}
=
T_{W\leftarrow I_t}T_{I\leftarrow G}.
$$

其中 $T_{I\leftarrow G}=T_{G\leftarrow I}^{-1}$ 仍然是每种车辆固定的刚体变换。历史帧应使用这些 pose 做正常的 SE(3) 对齐；这与逐帧把道路旋转成水平面完全不同。

为了保留真实下坡和容忍标定误差，voxel grid 在地面下方必须留有 margin。第一版可以从 $z_{min}=-0.5\sim-1.0\,\mathrm m$ 起步，再根据各距离段的地面高度分位数确定最终范围。

这一步预期主要改善多车型训练的收敛和跨 rig 泛化；如果所有数据本来已经在同一个 ground-centered ego frame，它只是一项坐标一致性检查，不应期待很大的额外精度收益。

### 5.2 网络张量流

设：

- Camera 数量为 $N$；
- 图像大小为 $H_I\times W_I$；
- BEV 大小为 $H\times W$；
- BEV feature channel 为 $C$；
- occupancy 高度层数为 $Z$。

```text
Images                        [B,N,3,H_I,W_I]
LiDAR                         variable-length points
   │                                      │
   ▼                                      ▼
Image Encoder                         Geometry Encoder
   │                                      │
   ▼                                      ▼
Image features                    Height-aware LiDAR BEV
[B,N,C_i,h_i,w_i]                    [B,C_l,H,W]
   │
   ▼
Depth-guided Lift/BEVPool
   │
   ▼
Camera BEV [B,C_c,H,W]
   │                                      │
   └──────── concat + 3×3 Conv2D ─────────┘
                         │
                         ▼
              Shared BEV [B,C,H,W]
                         │
       ┌─────────────────┼──────────────────┐
       ▼                 ▼                  ▼
[B,2,H,W]          [B,Z,H,W]        CenterHead outputs
ground μ/logσ      occupancy logits  heatmap/regression
```

整个主干在二维 BEV 上运行。只有 occupancy 最终 logits 带有 $Z$ 维，不维护：

$$
F_{3D}\in\mathbb R^{B\times C\times Z\times H\times W}
$$

这种昂贵的 dense 3D 中间 feature。

## 6. Camera branch

### 6.1 Image encoder

多相机使用共享的 image backbone 和 neck：

$$
F_n^{img}=E_{img}(I_n).
$$

建议：

- 使用大规模预训练初始化 image backbone；
- 第一版选择能够满足实时性的轻量 backbone；
- 大视觉模型主要用于初始化或蒸馏，不直接把完整大模型塞进在线系统；
- Camera 和 LiDAR 分支都需要独立单模态 baseline，便于确认融合增益来自互补信息而不是某一分支完全压倒另一分支。

### 6.2 LiDAR-guided depth lifting

将当前 LiDAR points 投影到各相机，得到 sparse depth：

$$
D_{lidar}^{n}(u,v).
$$

按照 EFFOcc 的思路，先用一个小型 geometry encoder 编码 sparse depth，再与 image feature 在 depth network 输入端融合：

$$
F_n^{img-depth}
=
\left[
F_n^{img};
E_d(D_{lidar}^{n})
\right].
$$

$$
\left(
P_n(d\mid u,v),
F_n^{ctx}(u,v)
\right)
=
D_{net}\left(F_n^{img-depth}\right).
$$

然后执行 LSS/BEVPool：

$$
F_{cam}
=
\operatorname{BEVPool}
\left(
P_n(d\mid u,v)F_n^{ctx}(u,v)
\right).
$$

这次早期交互使 LiDAR sparse depth 能直接改善：

- 图像 feature 的真实距离；
- 深度分布；
- BEV 落点位置。

其中 image feature 仍然提供主要的稠密外观和上下文，sparse depth 为 depth/context network 提供公制几何条件。

如果第一版工程复杂度需要降低，可以先使用普通 camera view transformer，把 projected LiDAR depth injection 作为第二阶段消融。

## 7. LiDAR geometry branch

LiDAR 分支不需要大型 3D semantic backbone。推荐使用：

- PointPillars；或
- SECOND-lite sparse encoder；或
- 按高度分 bin 的浅层 BEV encoder。

输入 feature 可以包含：

$$
(x,y,z,\text{intensity},\Delta x,\Delta y,\Delta z,\text{density}).
$$

其中 $(x,y,z)$ 已经位于第 5.1 节定义的 canonical ego frame。绝大部分有效点满足 $z\geq0$ 是预期现象，并不要求人为把 feature 再中心化到正负对称；但必须保留真实负值和地面下方的 voxel margin。若希望改善数值尺度，可以对送入 feature encoder 的 $z$ 另做 dataset-level standardization，但 voxel indexing 仍使用 canonical 几何坐标。

为了不在压缩 BEV 时完全丢失垂直信息，可以把少量高度 bin 沿 channel 保存：

```text
LiDAR points
     │
     ▼
Sparse voxel/pillar encoding
     │
     ▼
4–8 coarse height bins
     │ flatten height into channels
     ▼
F_lidar [B,C_l,H,W]
```

这里保存的是低维几何摘要，不是大型 dense 3D feature volume。

## 8. EFFOcc-style BEV feature fusion

Camera view transformer 和 LiDAR encoder 输出空间范围、分辨率一致的 BEV feature：

$$
F_{cam}\in\mathbb R^{B\times C_c\times H\times W},
$$

$$
F_{lidar}\in\mathbb R^{B\times C_l\times H\times W}.
$$

先沿 channel 拼接：

$$
F_{cat}
=
\left[F_{cam};F_{lidar}\right]
\in
\mathbb R^{B\times(C_c+C_l)\times H\times W}.
$$

然后使用轻量二维卷积融合并压缩 channel：

$$
F_{fused}
=
\operatorname{ReLU}
\left(
\operatorname{BN}
\left(
\operatorname{Conv}_{3\times3}(F_{cat})
\right)
\right).
$$

这与 EFFOcc 的核心融合原则一致：

1. LiDAR projected depth 先参与 Camera BEV lifting；
2. Camera BEV 和 LiDAR BEV 再进行 feature-level fusion；
3. 融合与后续 scene encoder 都使用 2D operators；
4. 不使用全局 cross-attention，也不使用 dense 3D fusion volume；
5. 不预先规定 Camera 或 LiDAR 永远主导，让 fused convolution 根据任务损失学习互补信息。

这里的 `EFFOcc-style` 表示沿用它的两阶段融合机制，并不是复刻完整 EFFOcc：原方法的 semantic occupancy head 被我们替换成 binary non-ground C2H，同时额外加入 ground-height 和 known-detection heads。

### 8.1 实现前提

Concat 之前必须保证：

- 两个 BEV 使用相同的 ego 坐标系；
- 所有车型先通过各自固定外参转换到同一个 ground-centered canonical ego 定义；
- $x/y$ 范围和 voxel size 完全一致；
- 数据增强对两个模态同步应用；
- 使用独立的 projection/norm 控制两个分支的 feature scale；
- 标定、时间戳和 ego-motion compensation 正确。

如果 feature scale 相差过大，可以先做：

$$
\widetilde F_{cam}=N_c(P_c(F_{cam})),
\qquad
\widetilde F_{lidar}=N_l(P_l(F_{lidar})),
$$

再进行：

$$
F_{fused}
=
\operatorname{Conv}_{3\times3}
\left(
[\widetilde F_{cam};\widetilde F_{lidar}]
\right).
$$

### 8.2 为什么第一版不使用 gate

`Camera-led gated residual` 没有直接的 occupancy 论文证据，而且 residual 写法本身也不能保证 Camera 真正主导。对于现在的 binary occupancy 任务，LiDAR return 本身就是非常有价值、类别无关的 OOD 几何证据，没有必要人为压低它。

因此第一版冻结为 concat + Conv2D。以下内容只作为后续消融，不属于默认网络：

- BEVFusion-style channel attention；
- reliability-aware fusion gate；
- Camera-led residual；
- deformable cross-attention。

只有这些模块在相同计算预算下明确优于 concat baseline，才升级默认设计。

## 9. Shared 2D BEV encoder

$$
F_{BEV}=E_{BEV}^{2D}(F_{fused}).
$$

推荐使用轻量 residual CNN 或 SECOND-style 2D neck。三个任务共享 BEV 上下文，但各自具有小型 branch-specific stem：

```text
F_BEV
  │
  ├── Ground stem → height
  ├── Occupancy stem → C2H
  └── Detection stem → CenterHead
```

这比三个任务完全共享最后一层更容易控制梯度冲突，额外成本很小。

---

# Part IV：三个学习 Head

## 10. Ground Height Head

建议输出：

$$
(\mu_g(x,y),s_g(x,y))
=
H_g(F_{BEV}),
$$

其中：

$$
\hat z_g=\mu_g,
\qquad
\sigma_g^2=\exp(s_g).
$$

Ground uncertainty 可以用于：

- 坡道和边缘区域降权；
- 下游增大安全余量；
- 过滤没有可靠 ground label 的 cell。

如果系统只追求最小实现，也可以先只输出一个 height channel：

```text
F_BEV → 3×3 Conv → 1×1 Conv → z_ground [B,1,H,W]
```

### 单层 ground 的限制

每个 $(x,y)$ 只预测一个 ground height，无法同时表示上下两层道路。第一版的定义是预测与 ego 当前道路连通的 ground；其他层保留在 non-ground occupancy 中。

如果多层道路成为主要失败模式，再升级为：

- 多峰 ground-height distribution；或
- ground surface proposals；或
- 3D ground occupancy。

第一版不需要提前承担这个复杂度。

## 11. Binary Channel-to-Height Occupancy Head

输入：

$$
F_{BEV}\in\mathbb R^{B\times C\times H\times W}.
$$

输出 $Z$ 个高度 logit：

$$
L_{ng}=H_{occ}(F_{BEV})
\in\mathbb R^{B\times Z\times H\times W}.
$$

转换成：

$$
p_{ng}(x,y,z)=\sigma(L_{ng}(x,y,z)).
$$

推荐第一版实现：

```text
F_BEV [B,C,H,W]
       │
       ▼
3×3 Conv2D + Norm + Activation
       │
       ▼
1×1 Conv2D: C_occ → Z
       │
       ▼
logits [B,Z,H,W]
       │ permute
       ▼
logits [B,H,W,Z]
```

这比原始 semantic FlashOcc 更轻，因为原始输出为 $Z\times K$，我们只输出 $Z$：

$$
\text{semantic C2H}: C\rightarrow ZK,
$$

$$
\text{our binary C2H}: C\rightarrow Z.
$$

### 为什么它能学习

它没有试图学习一个统一的 unknown 外观，而是在学习统一的物理属性：

$$
\text{non-ground matter/collision volume exists at }(x,y,z).
$$

Camera feature 提供边界和完整对象上下文，LiDAR 提供距离和高度，mapping/bbox GT 提供最终空间监督。因此 known、labeled unknown 和 latent unknown 不需要共享类别外观，也能共享 occupancy target。

### 可选的轻量纵向 refinement

如果 overhead 或多层结构表现不足，先在 logits 上增加 vertical Conv1D：

$$
L'(x,y,:)
=
L(x,y,:)
+
\operatorname{Conv1D}_z(L(x,y,:)).
$$

或者用 1–4 channel tiny 3D residual：

$$
L'=L+\operatorname{Conv3D}_{3\times3\times3}(L).
$$

它只处理低维 logits，不恢复大型 dense 3D feature。

## 12. Known-class Detection Head

使用 CenterPoint/CenterHead 风格输出：

```text
known-class heatmap
center offset
height
length / width / height
yaw
velocity
```

训练规则：

- known bbox：正常 detection positive；
- labeled unknown bbox：不作为 known 类 positive；
- labeled unknown bbox 覆盖区域最好从 detection classification negative loss 中 mask 掉，避免明确教模型“unknown object 就是背景”；
- 所有 known/unknown bbox 都可以进入 non-ground occupancy positive；
- 无 bbox static structure 对 detection 来说是普通背景，但对 occupancy 是 positive。

Detection head 不负责 occupancy 安全兜底。即使 detector 漏检，occupancy 仍应输出障碍。

---

# Part V：Observability Mask

## 13. 为什么需要它，但不把它变成预测类别

我们的 mapping GT 来自多帧 LiDAR，而模型输入是当前时刻的 Camera–LiDAR。Mapping 可能知道当前传感器视角完全看不到的结构。

如果把所有 map 中没有 occupied label 的位置都监督成 free，会出现：

- 未扫描区域被错误当成 free；
- 物体后方被错误当成 free；
- 网络因大量错误负样本而倾向预测全空；
- 当前帧被要求确定性恢复不可见结构。

Observability 解决的是 loss 有效范围，不增加 semantic 类别。

## 14. LiDAR ray mask

对于一条从 LiDAR origin 到 return point 的射线：

```text
LiDAR → observed free → observed free → hit voxel → unobserved behind hit
```

设射线穿过 voxel 集合为 $\mathcal R_i$，命中 voxel 为 $v_i^{hit}$：

$$
M_{lidar}(v)=1,
\quad
v\in\mathcal R_i\cup\{v_i^{hit}\}.
$$

回波后方保持 0，除非被其他射线或相机观察。

这可以用 voxel DDA/Bresenham ray traversal 实现，不需要训练参数。

## 15. Camera visibility mask

第一版可以只用 camera frustum：

$$
M_{FOV}(v)=\mathbf 1[\pi(v)\text{ 落入任一图像}].
$$

更准确的版本使用 depth z-buffer。设 voxel 深度为 $d_v$，可见表面深度为 $\hat d(u,v)$：

$$
M_{cam}(v)=
\begin{cases}
1,&d_v\leq \hat d(u,v)+\delta,\\
0,&d_v>\hat d(u,v)+\delta.
\end{cases}
$$

Camera 和 LiDAR 可以取并集：

$$
M_{obs}=M_{lidar}\lor M_{cam}.
$$

第一版最可靠的选择是只使用 LiDAR ray mask；camera visibility 作为后续增强。

## 16. 最终是否交给下游

当前讨论的第一版 planner 可以把低 occupancy 的 free 和 unobserved 都暂时当作可通过，因此 observability 不必成为强制下游输入。

但是建议内部仍保留它，用于：

1. occupancy loss mask；
2. 诊断“模型漏检”还是“当前根本没看到”；
3. 后续按 confirmed-free distance 限制车速；
4. 评估 scene completion 与直接感知的差别。

换句话说：

> Occupancy 网络只输出 binary probability；ignore 只存在于训练和可选的系统元数据中。

---

# Part VI：GT 构建

## 17. Ground GT

从 LiDAR mapping 的 ground surface 生成：

$$
z_g^*(x,y).
$$

需要同时生成：

- `ground_valid_mask`；
- 可选的 ground fitting residual/uncertainty；
- ego-connected ground component。

推荐通过 ego 所在 ground seed 做连通扩张，限制：

- 邻接高度差；
- 局部坡度；
- 法向；
- lane/route consistency。

未与当前道路连通的上层表面不应被 ground removal 错删。

## 18. Non-ground occupancy GT

定义：

$$
Y_{ng}
=
Y_{static\ map\ non-ground}
\cup
Y_{known\ bbox}
\cup
Y_{labeled\ unknown\ bbox}.
$$

### 18.1 Static map

从 static occupancy map 中移除当前 ego-connected ground：

$$
Y_{static\ ng}
=
Y_{static\ occ}\setminus Y_{ego-ground}.
$$

保留：

- 墙和建筑；
- vegetation；
- sign、pole、cone；
- 桥体和隧道顶棚；
- 与当前道路不连通的上层结构。

### 18.2 Bbox fill

所有 known 和 labeled unknown bbox 都填入 occupancy GT：

$$
Y_{box}(v)=\mathbf 1[v\in B_i].
$$

为了不抹掉车底下的 ground，可裁剪为：

$$
Y_{box}(x,y,z)=1
\quad\text{only if}\quad
z>z_g^*(x,y)+\delta_g.
$$

由于 bbox 内部被完全填充，这个 target 的含义是：

$$
\boxed{\text{conservative collision volume，而不是精确物体表面。}}
$$

这对规划是合理的，但评估时不能把它解释为精细 shape reconstruction。

### 18.3 Known 与 unknown 的标签流

```text
Known bbox
  ├── Occupancy GT: positive
  └── Detection GT: known class positive

Labeled unknown bbox
  ├── Occupancy GT: positive
  └── Detection GT: ignore, not background negative

Static object without bbox
  ├── Occupancy GT: positive
  └── Detection GT: no object target
```

## 19. Free 与 ignore

对于 binary non-ground head：

```text
non-ground occupied voxel       → y = 1
observed free voxel             → y = 0
ground voxel                    → y = 0 或低权重 negative
unobserved/ambiguous voxel      → ignore
```

最关键的规则：

$$
\boxed{
\text{Not in occupancy map}\neq\text{observed free}.
}
$$

可靠 free label 主要来自：

- LiDAR ray 在 return 前穿过的 voxel；
- 多帧 mapping 中反复被 ray traversed 的 voxel；
- 在精确 depth 和遮挡模型下确认的 camera free space。

## 20. 当前帧输入与多帧 map GT 的匹配

推荐第一版使用：

$$
M_{train}
=
M_{GT-valid}
\cap
M_{current-observable}.
$$

只在当前输入有能力判断的位置使用强监督。

如果后续明确要做 scene completion，可加入低权重不可见监督：

$$
\mathcal L_{occ}
=
\mathcal L_{visible}
+
\lambda_{completion}\mathcal L_{occluded},
\qquad
0<\lambda_{completion}<1.
$$

不要在第一版中把 direct perception 和 completion 混在一起而不做区分。

---

# Part VII：训练目标

## 21. Occupancy loss

使用 masked positive-weighted BCE：

$$
\mathcal L_{BCE}
=
-\frac{1}{\sum_v M_v}
\sum_v M_v
\left[
w_+y_v\log p_v
+
w_-(1-y_v)\log(1-p_v)
\right].
$$

再加入一个区域重叠损失：

$$
\mathcal L_{occ}
=
\lambda_{bce}\mathcal L_{BCE}
+
\lambda_{lovasz}\mathcal L_{Lovasz}.
$$

也可以用 Dice 替代 Lovász。第一版不要同时堆叠过多 loss。

由于 free voxel 数量远大于 occupied voxel，需要：

- positive weighting 或 focal loss；
- 距离分段采样/权重；
- 不让大面积 ground negative 淹没小障碍；
- 单独监控 occupied recall，而不是只看 accuracy。

## 22. Ground loss

基础版本：

$$
\mathcal L_{ground}
=
\frac{1}{\sum M_g}
\sum_{x,y}M_g(x,y)
\operatorname{SmoothL1}
\left(
\hat z_g,z_g^*
\right).
$$

如果输出 uncertainty：

$$
\mathcal L_{ground}
=
\frac{1}{\sum M_g}
\sum M_g
\left[
\exp(-s_g)|\hat z_g-z_g^*|
+s_g
\right].
$$

可加入低权重 gradient loss 保持坡面连续：

$$
\mathcal L_{slope}
=
\left\|
\nabla\hat z_g-\nabla z_g^*
\right\|_1.
$$

## 23. Detection loss

使用标准 CenterHead losses：

$$
\mathcal L_{det}
=
\mathcal L_{heatmap}
+
\mathcal L_{center}
+
\mathcal L_{size}
+
\mathcal L_{yaw}
+
\mathcal L_{velocity}.
$$

Labeled unknown bbox 区域在 known heatmap negative loss 中 mask 掉。

## 24. 可选的模态鲁棒性约束

EFFOcc-style concat/Conv2D 是默认融合，不强制规定哪种模态主导。为了诊断模型是否完全忽略 Camera，可以在消融实验中从 $F_{cam}$ 接一个轻量 camera-only auxiliary head：

$$
p_{ng}^{cam}=H_{occ}^{aux}(F_{cam}).
$$

$$
\mathcal L_{cam-aux}
=
\operatorname{BCE}
\left(
p_{ng}^{cam},Y_{ng}
\right).
$$

该 head 推理时删除。还可以消融随机 LiDAR dropout 和点云稀疏度扰动。

这些不是默认训练所必需的模块。只有在实验显示 fused model 只会复制 LiDAR hits、不能利用 Camera 做稠密补全时再启用。

## 25. 总损失

$$
\mathcal L_{main}
=
\lambda_{occ}\mathcal L_{occ}
+
\lambda_g\mathcal L_{ground}
+
\lambda_s\mathcal L_{slope}
+
\lambda_{det}\mathcal L_{det}.
$$

只有启用可选 camera auxiliary 时，才加入：

$$
\mathcal L
=
\mathcal L_{main}
+
\lambda_{cam}\mathcal L_{cam-aux}.
$$

推荐先把三个主任务独立训练或逐项开启，确认每个 GT 和 loss 正确，再联合调权重。默认先使用 $\mathcal L_{main}$。

---

# Part VIII：推理输出与下游

## 26. 感知模型输出

```python
PerceptionOutput = {
    "ground_height":             Tensor[B, H, W],
    "ground_uncertainty":        Tensor[B, H, W],       # optional
    "non_ground_occupancy_prob": Tensor[B, H, W, Z],
    "known_objects":             List[Box3D],
    "observability":             Tensor[B, H, W, Z],    # geometric, optional downstream
}
```

Occupancy 保留 probability，不在模型内部过早做 hard threshold。

## 27. 生成 vehicle-specific 2D collision BEV

设车辆碰撞高度范围为：

$$
z_{low}(x,y)
=
\hat z_g(x,y)+h_{min},
$$

$$
z_{high}(x,y)
=
\hat z_g(x,y)+h_{vehicle}+m_h.
$$

第一版可以从地面上方很小的 $\delta$ 开始，以保守捕获路沿和低矮掉落物：

$$
h_{min}=\delta.
$$

垂直压缩：

$$
p_{block}(x,y)
=
\max_{z\in[z_{low},z_{high}]}
p_{ng}(x,y,z).
$$

这一步会：

- 保留车辆高度范围内的 car、wall、debris；
- 排除高于车顶安全余量的 bridge/sign；
- 根据坡道变化动态移动 clipping window；
- 允许同一 3D occupancy 服务不同高度的车辆。

## 28. 加入车辆 footprint

车辆不是一个点。设车辆 footprint 和安全 margin 为 $\mathcal F$：

$$
C_{vehicle}(x,y)
=
\max_{(\Delta x,\Delta y)\in\mathcal F}
p_{block}(x+\Delta x,y+\Delta y).
$$

这等价于对障碍区域做车辆形状的 Minkowski sum。对于有朝向的轨迹，应使用旋转后的 footprint 或直接检查轨迹 swept volume。

最终给 planner：

```python
PlanningInput = {
    "collision_probability": Tensor[H, W],
    "hard_obstacle_mask":    Tensor[H, W],
    "ground_height":         Tensor[H, W],
    "known_tracks":          List[Track],
    "unknown_components":    List[UnknownObstacle],  # optional
}
```

## 29. Free 和 unobserved 的第一版策略

第一版下游可以使用：

$$
\text{traversable}(x,y)
=
\mathbf 1[p_{block}(x,y)<\tau].
$$

即暂时把低 occupancy 的 observed free 和 unobserved 采取相同通行策略，并通过高频重规划在靠近后更新。

这一策略成立的安全条件是：

$$
d_{confirmed/detectable}
>
d_{stop},
$$

$$
d_{stop}
=
v\,t_{latency}
+
\frac{v^2}{2a_{brake}}
+
d_{margin}.
$$

因此更稳健的系统分工是：

- path planner 可以允许通过 unobserved；
- speed planner 根据确认可见距离限制速度；
- 当前第一版如果暂不使用 observability，也应保留它用于日志和未来升级。

## 30. Known 与 residual unknown

将 known boxes rasterize 成 soft explanation mask：

$$
M_{known}(v)
=
\max_i s_i\mathbf 1[v\in B_i].
$$

Residual unknown：

$$
p_{unknown}(v)
=
p_{ng}(v)
\left(1-M_{known}(v)\right).
$$

注意：

$$
\boxed{
\text{Planner 始终使用 total occupancy，不能只使用 residual unknown。}
}
$$

Known explanation 只用于：

- 给 occupancy 添加类别和速度属性；
- 把无法解释的 occupied region 标记为 potential OOD；
- unknown component clustering/tracking；
- 数据挖掘与主动学习。

它不能从总碰撞图中删除 known object 的 occupancy。

## 31. Unknown component 后处理

如果需要把 residual occupancy 暂时转成 object：

```text
Residual unknown voxels
          │
          ▼
Threshold + 3D connected components / DBSCAN
          │
          ▼
centroid / size / nearest distance / temporary bbox
          │
          ▼
temporal association + velocity estimate
```

这个 bbox 是规划和日志用途的后处理结果，不要求主网络重新引入 entity query。

---

# Part IX：效率设计

## 32. 为什么这条路线适合 20 FPS

计算主要为：

```text
Image backbone
+ camera view transformation
+ lightweight LiDAR encoder
+ 2D BEV fusion/encoder
+ three lightweight heads
```

没有：

- 多层 dense 3D convolution；
- 大规模 3D transformer；
- hundreds of Gaussian/query iterative updates；
- semantic $Z\times K$ occupancy classifier。

Occupancy head 只输出：

$$
H\times W\times Z
$$

个 logit，而不是：

$$
H\times W\times Z\times K.
$$

## 33. 推荐起始配置

以下是第一轮 profiling 的起点，不是必须参数：

| 模块 | 推荐起点 |
|---|---|
| Camera | 6 views |
| Image input | $256\times704$ |
| Image backbone | 轻量 pretrained backbone，先以 R18 级别成本为基准 |
| BEV resolution | $0.4$ m |
| BEV channel | 96 或 128 |
| Height bins | $Z=16$ 起步，按实际高度范围确定 |
| LiDAR | PointPillars/SECOND-lite，4–8 coarse height bins |
| Coordinate preprocessing | per-rig fixed transform 到 ground-centered ego；保留下坡/坑洼负高度，不做逐帧 ground flattening |
| Fusion | projected LiDAR depth guidance + BEV concat + $3\times3$ Conv2D/BN/ReLU |
| Occupancy | $3\times3$ Conv + $1\times1$ Conv 输出 $Z$ |
| Detection | lightweight CenterHead |
| Precision | FP16 |

必须在目标硬件测量完整延迟：

```text
data preprocessing
+ image backbone
+ projected depth / BEVPool
+ LiDAR voxelization
+ LiDAR encoder
+ BEV fusion and encoder
+ all heads
+ collision adapter
+ detection post-processing
```

不要把论文中 TensorRT、FP16、特定 GPU 的 FPS 与本地 PyTorch 端到端速度直接比较。

---

# Part X：实验与验收

## 34. 第一阶段：先验证标签，而不是先调大模型

在训练前可视化至少 100 个场景：

1. 每种车型转换后的近场 ground-$z$ 分布；
2. 不同距离区间的 ground-$z$ 分布，确认没有把坡道或负高度截断；
3. canonical LiDAR 到各 Camera 的重投影对齐；
4. current ego-connected ground；
5. ground-removed static occupancy；
6. known bbox fill；
7. unknown bbox fill；
8. LiDAR ray free voxels；
9. ignored/unobserved voxels；
10. 车辆高度 clipping 后的 2D collision GT；
11. bridge/tunnel/overhead cases。

如果标签定义不一致，更换 backbone 不会解决问题。

## 35. 关键模型消融

在模型消融之前，先固定一组坐标预处理消融，用来验证跨车型 normalization 的实际收益：

| ID | 坐标设置 |
|---|---|
| C0 | 各车型保留原始 sensor-$z$ 定义 |
| C1 | 按车型固定外参转换到 ground-centered canonical ego，默认方案 |

C0/C1 保持训练样本、模型和损失完全一致，分别报告单车型、跨车型和 held-out rig 性能。重点观察收敛速度、各高度段 occupancy recall、远处地面 false-positive，以及不同车型之间的性能方差。

| ID | 模型 |
|---|---|
| A0 | Camera-only BEV + binary C2H |
| A1 | LiDAR-only BEV + binary C2H |
| A2 | Camera/LiDAR BEV concat + $3\times3$ Conv2D，不注入 sparse depth |
| A3 | A2 + projected LiDAR depth guidance，完整 EFFOcc-style 两阶段融合 |
| A4 | A3 + LiDAR coarse height-bin features |
| A5 | A4 + vertical Conv1D refinement |
| A6 | A3 + adaptive/reliability gate，研究扩展 |

A3 是默认最终融合。A0/A1 用于确认两个单模态分支的能力，A2/A3 分离 early depth guidance 与 BEV feature fusion 的贡献。A6 只有明确优于 A3，才有理由取代 concat/Conv2D。

## 36. Head 和监督消融

| ID | 设置 |
|---|---|
| B0 | BCE only |
| B1 | weighted BCE + Lovász |
| B2 | B1 + current observability mask |
| B3 | B2 + LiDAR modality dropout |
| B4 | B2 + camera-only auxiliary loss |
| B5 | B2 + visible supervision + low-weight completion supervision |

## 37. 核心指标

### Ground

- height MAE/RMSE；
- slope error；
- road-edge error；
- 坡道和桥下单独统计。

### Occupancy

- binary IoU；
- occupied precision/recall；
- free-space false blockage；
- 距离分段 recall；
- 高度分段 recall；
- 小障碍 recall。

### OOD 重点拆分

- known bbox 内 occupancy recall；
- labeled unknown bbox 内 occupancy recall；
- 无 bbox static obstacle recall；
- 完全未在训练中出现的 latent unknown recall；
- residual unknown component recall/false positive。

### Downstream

- collision voxel recall；
- collision BEV recall；
- false blocked area；
- overhead false collision rate；
- trajectory collision miss rate；
- 可停止距离内的障碍 recall。

### 系统

- end-to-end FPS；
- P50/P95 latency；
- peak GPU memory；
- Camera-only、LiDAR-only 和 fusion robustness。

## 38. 必测场景

- 车辆和行人；
- labeled unknown boxes；
- 没有 bbox 的 sign/cone/pole；
- 路侧墙和 vegetation；
- 掉落物和小障碍；
- bridge/overpass；
- tunnel；
- 坡道；
- 远距离；
- 强遮挡；
- LiDAR 稀疏/雨雾退化；
- camera 过曝、暗光和模糊。

---

# Part XI：风险与升级路径

## 39. 主要风险

### 39.1 Mapping GT 与当前输入不匹配

症状：可见区域表现尚可，不可见区域大量不稳定预测。  
处理：current observability mask；明确区分 direct perception 和 completion。

### 39.2 Ground removal 错删 overhead structure

症状：桥面、隧道顶棚在 GT 中消失。  
处理：ground 定义为 ego-connected support surface，不是所有水平面。

### 39.3 Filled bbox 与 static surface GT 定义不同

症状：动态物体预测成实心块，静态结构只预测表面。  
处理：明确任务是 conservative collision occupancy；必要时对 static surface 做安全厚度膨胀，并按该定义评估。

### 39.4 Occupancy 只复制 LiDAR hits

症状：Camera 遮挡补全和稀疏区域没有提升。  
处理：先对比 A0/A1/A3；再尝试 camera-only auxiliary head、LiDAR dropout 和稀疏度扰动，不先修改默认融合结构。

### 39.5 C2H 缺乏垂直推理

症状：桥下 free gap、细杆和多层结构错误。  
处理顺序：vertical Conv1D → tiny logit 3D refinement → 最后才考虑 dense 3D decoder。

### 39.6 Detector 漏检造成大量 residual unknown

症状：普通 car 经常被标为 unknown。  
处理：unknown residual 只用于解释，不改变 total occupancy；对 known explanation 使用 soft confidence；提高 known detector recall。

## 40. 升级顺序

```text
Stage 0  标签与 collision adapter 验证
   ↓
Stage 1  Camera-only binary C2H + ground
   ↓
Stage 2  Camera/LiDAR BEV concat + 3×3 Conv2D
   ↓
Stage 3  加入 projected LiDAR depth guidance，形成完整 EFFOcc-style fusion
   ↓
Stage 4  Joint known detection + residual unknown analysis
   ↓
Stage 5  Vertical refinement / temporal tracking
   ↓
Stage 6  只有明确需要时再研究 adaptive gate、completion、entity 或 future occupancy
```

---

# Part XII：最终接口和一句话定义

## 41. 最小模型接口

```python
class FinalOODOccupancyModel:
    def forward(self, images, lidar, calibration, rig_id):
        # Original calibration maps every sensor to the common IMU/ego frame I.
        # T_G_from_I is a fixed per-rig origin shift; it is not frame-wise
        # pitch/roll compensation and does not flatten the observed road.
        T_G_from_I = self.fixed_ground_from_imu[rig_id]
        T_G_from_L = T_G_from_I @ calibration.T_I_from_L
        T_G_from_C = [
            T_G_from_I @ T_I_from_C_i
            for T_I_from_C_i in calibration.T_I_from_C
        ]

        lidar_g = transform_points(lidar, T_G_from_L)
        calibration_g = calibration.with_target_frame(
            lidar_extrinsic=T_G_from_L,
            camera_extrinsics=T_G_from_C,
        )

        sparse_depth = project_lidar_to_cameras(lidar_g, calibration_g)
        f_cam = self.camera_bev(images, sparse_depth, calibration_g)
        f_lidar = self.lidar_geometry_bev(lidar_g)

        f_cam = self.camera_projection_norm(f_cam)
        f_lidar = self.lidar_projection_norm(f_lidar)
        f_fused = self.fusion_conv(concat([f_cam, f_lidar], dim=1))
        f_bev = self.bev_encoder(f_fused)

        ground_height = self.ground_head(f_bev)
        non_ground_logits = self.binary_c2h_head(f_bev)
        known_objects = self.known_detection_head(f_bev)

        return {
            "ground_height": ground_height,
            "non_ground_occupancy_prob": sigmoid(non_ground_logits),
            "known_objects": known_objects,
        }
```

Observability 在模型外计算：

```python
# 使用与网络相同的 canonical LiDAR，而不是原始 sensor-frame points。
lidar_g = transform_points(lidar, fixed_rig_transform[rig_id])
observability = voxel_ray_tracing(lidar_g, canonical_voxel_grid)
```

下游 adapter：

```python
collision_bev = vertical_clip_and_max(
    occupancy=non_ground_occupancy_prob,
    ground_height=ground_height,
    vehicle_bottom=h_min,
    vehicle_top=h_vehicle + height_margin,
)

collision_cost = footprint_inflation(
    collision_bev,
    vehicle_footprint,
    safety_margin,
)
```

## 42. 最终数学定义

感知：

$$
(I_{1:N},T_{G\leftarrow L_v}P)
\rightarrow
\left(
\hat z_g(x,y),
p_{ng}(x,y,z),
\mathcal B_{known}
\right).
$$

规划适配：

$$
p_{block}(x,y)
=
\max_{z\in\mathcal H_{vehicle}(\hat z_g)}
p_{ng}(x,y,z).
$$

开放世界解释：

$$
p_{unknown}(v)
=
p_{ng}(v)
\left(1-M_{known}(v)\right).
$$

一句话概括最终方法：

> **先按每种车辆的固定传感器外参，将 Camera、LiDAR 和全部标签统一到车辆附近地面约为 $z=0$ 的 canonical ego frame，同时保留真实坡道和负高度；随后采用 EFFOcc-style 两阶段融合：先用 projected LiDAR sparse depth 改善 Camera lifting，再将 Camera BEV 与 LiDAR BEV concat 并用 $3\times3$ Conv2D 融合；通过 ground-height head 和 FlashOcc-style binary channel-to-height head 预测当前地面与所有非地面占用，再由 known-only detection 为部分 occupancy 添加类别和运动属性，无法被解释的 residual occupancy 即为潜在 OOD 障碍；规划端按照车辆相对地面高度将 3D occupancy 裁剪成实时 2D collision BEV。**

---

## 参考设计来源

- [FlashOcc：Channel-to-Height occupancy](https://arxiv.org/abs/2311.12058)
- [FlashOcc 官方代码](https://github.com/Yzichen/FlashOCC)
- [EFFOcc：高效 Camera–LiDAR BEV fusion](https://arxiv.org/abs/2406.07042)
