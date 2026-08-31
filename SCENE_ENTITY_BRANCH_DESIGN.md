# Scene Branch + Entity Branch：面向 Static Map 与 3D BBox 标注的 OOD Occupancy 设计

> 适用数据形式：  
> 1. 使用 LiDAR mapping 构建的静态场景 occupancy map；  
> 2. 前景物体只有 3D bounding box 标注；  
> 3. 当前 object occupancy 定义为 bbox 内部全部填满，而非物体真实表面。  
>
> 相关文档：[SAM 3D Objects 与 OOD Occupancy](./SAM3D_FOR_OOD_OCCUPANCY.md)

---

## 0. 核心结论

这种标注形式非常适合把 occupancy 显式分解为：

$$
Y_t^{\mathrm{total}}
=
Y^{\mathrm{static\ scene}}
\cup
\bigcup_k Y_{t,k}^{\mathrm{bbox}}.
$$

其中：

- **Scene Branch** 由 LiDAR mapping 得到的 static occupancy map 监督；
- **Entity Branch** 由当前帧的 3D bbox 监督；
- Entity Branch 第一版不需要预测精细物体形状，只需预测 bbox、objectness、semantic/OOD 属性；
- 通过可微 box rasterizer 将 bbox 转成 filled-box occupancy；
- Scene 与 Entity 的区别不应完全让网络自行发现，而应利用标注来源显式构造 voxel ownership；
- 推理时使用置信度控制的 Entity 优先合成，而不是简单相加或无条件清空 Scene。

第一版模型可以概括为：

```text
Camera + LiDAR
      │
      ▼
Shared / Asymmetric Fusion Backbone
      │
      ├──────────────────────────────┐
      ▼                              ▼
Scene Branch                    Entity Branch
static occupancy                objectness + 3D bbox
                                      │
                                      ▼
                              Soft Box Rasterizer
                                      │
      └──────────────┬───────────────┘
                     ▼
              Ownership Composer
                     │
                     ▼
           Final Semantic/OOD Occupancy
```

---

## 1. 为什么这个数据集天然适合双分支？

数据集的标注构建过程本身已经隐式地定义了两种不同来源：

```text
长期、多帧 LiDAR mapping
          │
          ▼
static scene occupancy

当前时刻的 object annotations
          │
          ▼
3D bounding boxes
```

两种标签表达的对象不同：

| 标签来源 | 表达内容 | 适合的模型分支 |
|---|---|---|
| LiDAR mapping | 道路、墙、建筑、路缘、长期静态结构 | Scene Branch |
| 3D bbox | 当前时刻独立物体的位置、尺寸、朝向和类别 | Entity Branch |

因此没有必要让一个 dense occupancy head 同时学习：

- 静态场景几何；
- 物体发现；
- 物体实例划分；
- bbox 填充；
- known/OOD semantic 判断。

将它们分开后，每个分支都可以直接利用最匹配的监督。

---

## 2. 首先明确：Scene 与 Entity 按什么规则划分？

这里存在两种可能的定义，必须在构造标签前选定。

### 2.1 定义 A：当前帧互斥 Ownership

对当前帧的每一个 occupied voxel，只允许它属于 Scene 或 Entity 中的一方：

$$
z(x)\in
\{\mathrm{free},\mathrm{scene},\mathrm{entity},\mathrm{ignore}\}.
$$

如果 voxel 位于某个 bbox 内，则 Entity 优先；否则，如果 static map 显示 occupied，则属于 Scene。

这种定义最适合当前任务，因为 object occupancy 本来就是 filled box。

优点：

- 标签定义明确；
- 两个分支容易单独监督；
- 可以抑制分支职责塌缩；
- 最终输出与现有评测定义一致。

缺点：

- bbox 内的永久静态表面，例如车辆下方的 road，可能从当前 Scene target 中被移除；
- Scene Branch 输出不再是严格意义上的完整 amodal static map。

### 2.2 定义 B：Amodal Static Map + Current Entities

Scene Branch 始终预测永久静态场景，包括当前被物体遮挡的 road 或其他结构；Entity Branch 额外预测当前物体。

$$
Y_t^{\mathrm{total}}
=
Y^{\mathrm{amodal\ static}}
\cup
Y_t^{\mathrm{entity}}.
$$

这种定义允许 Scene 和 Entity 在同一 voxel 上重叠，不再强制 ownership 互斥。

优点：

- 保留完整静态地图；
- 适合地图重建和长期世界模型；
- 物体离开后不需要重新恢复被遮挡的路面。

缺点：

- 训练和融合更复杂；
- 全局 overlap penalty 不再成立；
- 当前 filled-box entity 和 road 等静态结构会发生大范围重叠；
- Scene/Entity attribution 的评测需要另行定义。

### 2.3 第一版推荐

第一版建议使用 **定义 A：当前帧互斥 Ownership**。

原因不是它在物理上最完整，而是它最符合当前标注和评测：

> 只要 voxel 位于一个 GT bbox 内，就由 Entity Branch 负责；bbox 以外的 static occupancy 由 Scene Branch 负责。

等该 baseline 建立后，可以再额外增加一个 amodal-static-map head，而不是一开始混合两种定义。

---

## 3. 怎样从 static map 和 bbox 构造监督？

设：

- $Y_{\mathrm{map}}(x)\in\{0,1\}$：LiDAR mapping 构建的 static occupancy；
- $B_k$：当前帧第 $k$ 个 oriented 3D bbox；
- $x$：一个 voxel center。

### 3.1 Entity mask

将所有 GT bbox 栅格化：

$$
M_E(x)
=
\max_k\mathbf 1[x\in B_k].
$$

由于当前 object occupancy 定义为 bbox 内部全部填满，因此：

$$
Y_E(x)=M_E(x).
$$

如果多个 bbox 重叠，可以使用 instance ownership：

$$
k^*(x)
=
\arg\max_k
\left[
\mathbf 1[x\in B_k]w_k(x)
\right],
$$

其中 $w_k(x)$ 可以根据 bbox confidence、voxel 到 box center 的距离或标注优先级定义。

### 3.2 Scene target

在互斥定义下：

$$
Y_S(x)
=
Y_{\mathrm{map}}(x)
\left(1-M_E(x)\right).
$$

即使传感器在该位置确实看到物体，只要位置属于 bbox，Scene Branch 的目标就是零，因为该 voxel 的 ownership 已经分给 Entity。

### 3.3 Total occupancy

最终 GT occupancy 为：

$$
Y_{\mathrm{total}}(x)
=
Y_E(x)
+
\left(1-Y_E(x)\right)Y_S(x).
$$

由于 $Y_E$ 和 $Y_S$ 被定义成互斥，它也等价于二者的 union。

### 3.4 Ownership target

$$
z(x)=
\begin{cases}
\mathrm{entity}, & M_E(x)=1,\\
\mathrm{scene}, & M_E(x)=0,\ Y_{\mathrm{map}}(x)=1,\\
\mathrm{free}, & M_E(x)=0,\ Y_{\mathrm{map}}(x)=0,\\
\mathrm{ignore}, & x\in\Omega_{\mathrm{ambiguous}}.
\end{cases}
$$

其中 $\Omega_{\mathrm{ambiguous}}$ 包括：

- bbox 边界附近；
- 标定或时间同步不确定区域；
- static map 与 bbox 严重冲突区域；
- 缺失标注或难以归属的 voxel。

---

## 4. BBox 监督是否足以学出 Entity？

### 4.1 对当前 object occupancy 定义而言，基本足够

Entity Branch 可以直接作为一个 query-based 3D detector：

```text
Visual entity queries
        │
        │ query LiDAR geometric features
        ▼
3D bbox + objectness + class/OOD
        │
        ▼
Differentiable Box Rasterizer
        │
        ▼
Filled Entity Occupancy
```

每个 entity query 输出：

$$
\hat e_k
=
\left(
\hat B_k,
\hat q_k^{\mathrm{entity}},
\hat c_k,
\hat s_k^{\mathrm{OOD}},
\hat f_k
\right),
$$

其中：

- $\hat B_k$：3D bbox；
- $\hat q_k^{\mathrm{entity}}$：class-agnostic entity/objectness confidence；
- $\hat c_k$：known class distribution；
- $\hat s_k^{\mathrm{OOD}}$：entity-level OOD score；
- $\hat f_k$：entity feature，可用于 temporal tracking 和 open-vocabulary matching。

3D bbox 参数为：

$$
\hat B_k
=
(x_k,y_k,z_k,l_k,w_k,h_k,\theta_k).
$$

### 4.2 BBox 能监督什么？

BBox 足以监督：

- 是否存在一个独立 entity；
- entity center；
- metric extent；
- yaw/orientation；
- instance 数量与区分；
- filled-box occupancy；
- known class；
- temporal association（如果存在 track ID）。

### 4.3 BBox 不能监督什么？

BBox 无法直接监督：

- 物体真实表面；
- bbox 内的自由空间；
- 精细 instance occupancy；
- object mask 或 object Gaussian 的局部结构；
- 不可见部分的真实形状。

但这些并不是当前 object occupancy label 所要求的内容。因此第一版不应增加一个缺乏直接监督的复杂 shape decoder。

---

## 5. Entity Occupancy：直接使用可微 Box Rasterizer

对预测 bbox $\hat B_k$，构造 soft inside probability：

$$
\hat M_k(x)
=
\operatorname{SoftBoxRasterize}(x,\hat B_k;\tau),
$$

其中 $\tau$ 控制 box boundary 的软硬程度。

一种抽象实现方式是：先把 voxel center 变换到 box local coordinates：

$$
x_k^{\mathrm{local}}
=
R(-\theta_k)(x-t_k),
$$

再计算它到各个 box face 的 normalized margin：

$$
d_k(x)
=
1-
\max
\left(
\frac{2|x_{k,x}^{\mathrm{local}}|}{l_k},
\frac{2|x_{k,y}^{\mathrm{local}}|}{w_k},
\frac{2|x_{k,z}^{\mathrm{local}}|}{h_k}
\right).
$$

使用 sigmoid 得到 soft mask：

$$
\hat M_k(x)
=
\sigma\left(\frac{d_k(x)}{\tau}\right).
$$

结合 entity confidence：

$$
p_{E,k}(x)
=
\hat q_k^{\mathrm{entity}}\hat M_k(x).
$$

多个 entity 的 union：

$$
p_E(x)
=
1-
\prod_k
\left(1-p_{E,k}(x)\right).
$$

推理时可以改用 hard oriented-box rasterization，以严格匹配当前评测标签。

---

## 6. Objectness、Known Class 和 OOD 必须解耦

不应该使用：

$$
p(\mathrm{entity})
=
\max_{c\in\mathcal C_{\mathrm{known}}}p(c).
$$

否则，真正的 OOD object 因为所有 known-class confidence 都低，会被错误当成 background。

应该显式分解为：

$$
p(\mathrm{entity}),
\qquad
p(c\mid\mathrm{entity}),
\qquad
p(\mathrm{OOD}\mid\mathrm{entity}).
$$

三个问题分别是：

1. 这里是否存在一个独立物体？
2. 如果存在，它最像哪个已知类别？
3. 它是否超出了已知类别分布？

Entity detector 的基本损失为：

$$
\mathcal L_{\mathrm{det}}
=
\lambda_{\mathrm{obj}}\mathcal L_{\mathrm{objectness}}
+
\lambda_{\mathrm{box}}\mathcal L_{\mathrm{bbox}}
+
\lambda_{\mathrm{cls}}\mathcal L_{\mathrm{known-class}}.
$$

其中 objectness 必须是 class-agnostic，并由“是否匹配到任何 GT bbox”监督，而不是由 known class 最大概率间接产生。

### 6.1 BBox 监督对 OOD 的局限

如果训练数据只包含已知类别 bbox，那么 bbox-supervised objectness 可以学到一定的 category-general objectness，但不能保证发现任意未知物体。

后续可以加入：

- SAM/SAM 3 class-agnostic mask proposals；
- 大规模预训练视觉特征；
- box-supervised mask/objectness pretraining；
- 当前传感器观测与 static scene prediction 的 residual proposals；
- 未标注物体的 pseudo boxes；
- known object feature perturbation 或 synthetic OOD entities。

但这些属于第二阶段。第一版应先验证 class-agnostic bbox query 是否已经优于 voxel-wise OOD。

---

## 7. Scene Branch 为什么会预测出 Entity？

即使 Scene Branch 的目标只是 static scene，它依然能从 camera 和 LiDAR features 中看到汽车、行人等当前物体。

如果只监督最终 occupancy：

$$
\mathcal L
\left(
p_S\cup p_E,
Y_{\mathrm{total}}
\right),
$$

网络存在多个等价解：

```text
解 1：Scene Branch 预测所有 occupancy，Entity Branch 被忽略

解 2：Entity Branch 预测很多区域，Scene Branch 被削弱

解 3：两个 Branch 重复预测相同物体
```

这就是 branch responsibility collapse。

因此只监督 final union 是不够的。必须同时提供：

- Scene-specific target；
- Entity-specific target；
- ownership 或 overlap 约束；
- final occupancy consistency。

---

## 8. 怎样阻止 Scene Branch 吸收 Entity？

### 8.1 分支独立监督

Scene Branch 使用：

$$
\mathcal L_{\mathrm{scene}}
=
\operatorname{BCE}
\left(p_S,Y_S;M_{\mathrm{valid}}^S\right).
$$

在互斥 ownership 定义下，bbox 内部的 Scene target 为零。因此 Scene Branch 会获得明确的负监督：

> 这些位置虽然在当前传感器中表现为 occupied，但它们不属于 static Scene Branch。

Entity occupancy 使用：

$$
\mathcal L_{\mathrm{entity-occ}}
=
\operatorname{BCE}
\left(p_E,Y_E;M_{\mathrm{valid}}^E\right).
$$

### 8.2 Branch overlap penalty

在 ownership 明确的 voxel 上，可以加入：

$$
\mathcal L_{\mathrm{overlap}}
=
\frac{1}{|\Omega_{\mathrm{exclusive}}|}
\sum_{x\in\Omega_{\mathrm{exclusive}}}
p_S(x)p_E(x).
$$

它惩罚两个分支同时 claim 同一个 voxel。

需要注意：

- 不要在 bbox boundary 的 ignore band 上计算；
- 不要对 amodal-map 定义使用全局 overlap penalty；
- 权重不宜过大，否则 false-positive entity 可能压制真实 Scene。

### 8.3 显式 Ownership Head

还可以增加：

$$
p_z(x)
=
\operatorname{Softmax}
\left(h_{\mathrm{ownership}}(F(x))\right),
$$

其中：

$$
p_z(x)
=
\left[
p_{\mathrm{free}}(x),
p_{\mathrm{scene}}^{\mathrm{owner}}(x),
p_{\mathrm{entity}}^{\mathrm{owner}}(x)
\right].
$$

监督：

$$
\mathcal L_{\mathrm{ownership}}
=
\operatorname{CE}(p_z,z).
$$

不过第一版不一定需要独立 head。可以直接将 rasterized entity confidence 当作 ownership gate，减少系统复杂度。

---

## 9. 推理时怎样抑制 Scene 中的 Entity？

### 9.1 不建议无条件硬清除

不应该简单执行：

```python
scene_occupancy[predicted_entity_box] = 0
```

如果 bbox 是 false positive，这会把墙、路缘或其他真实场景结构清除掉。

### 9.2 Confidence-gated Entity ownership

由所有预测 bbox 构造 entity gate：

$$
g_E(x)
=
\max_k
\left[
\hat q_k^{\mathrm{entity}}\hat M_k(x)
\right].
$$

Scene 的有效输出为：

$$
p_S^{\mathrm{effective}}(x)
=
\left(1-g_E(x)\right)p_S(x).
$$

最终 occupancy：

$$
p_{\mathrm{final}}(x)
=
1-
\left(1-p_E(x)\right)
\left(1-p_S^{\mathrm{effective}}(x)\right).
$$

直觉上：

```text
高置信度 bbox
     │
     ├── Entity 获得该区域 ownership
     └── Scene 被显著抑制

低置信度 bbox
     │
     ├── Entity 只获得部分 ownership
     └── Scene 仍然保留，避免误检擦除真实场景
```

### 9.3 Semantic composition

如果 $g_E(x)$ 较高：

- semantic label 来自对应 entity query；
- 使用 entity-level known/OOD score；
- voxel 共享 entity identity。

如果 $g_E(x)$ 较低：

- semantic label 来自 Scene Branch；
- 使用 road、building、vegetation 等 scene/stuff 类别。

可以写成：

$$
p_{\mathrm{sem}}(x)
=
g_E(x)p_{\mathrm{sem}}^E(x)
+
\left(1-g_E(x)\right)p_{\mathrm{sem}}^S(x).
$$

---

## 10. BBox 边界和地面重叠怎样处理？

车辆 bbox 的底部可能与 road occupancy 相交：

```text
          Vehicle bbox
        ┌─────────────┐
        │█████████████│
────────┴─────────────┴──────── road occupancy
```

若直接把整个 bbox 当作 Scene negative，会把车辆下方的 road 从 Scene target 中移除。

在第一版互斥定义中，这可以接受，因为最终 occupancy 并不会丢失：该区域已经被 Entity filled box 占据。但为了减少标签噪声，建议：

1. 使用 oriented 3D bbox，而不是 axis-aligned bbox；
2. 检查 bbox bottom 与 ground voxel 的对齐；
3. 在 bbox 边缘构造 ignore band；
4. 对最底部一层 voxel 单独统计和处理；
5. 不在 ambiguous boundary 上计算 overlap loss。

一种 ignore band 是：

$$
M_{\mathrm{ignore}}
=
\operatorname{Dilate}(M_E,r_d)
-
\operatorname{Erode}(M_E,r_e).
$$

对应 voxel 不参与分支 ownership loss，但仍可参与较弱的 final occupancy loss。

---

## 11. 训练前必须做 Static Map–BBox 冲突审计

需要统计：

$$
r_{\mathrm{overlap}}
=
\frac{
|Y_{\mathrm{map}}\cap M_E|
}{
|M_E|
}.
$$

建议分别统计：

- 每种 object class；
- moving 与 stationary objects；
- bbox 最底部 1～2 个 voxel layers；
- bbox 中部和上部；
- 不同场景与不同 mapping sequence；
- bbox 膨胀/腐蚀前后的 overlap。

如果 bbox 中上部与 static map 的 overlap 很高，可能意味着：

- mapping 没有清除动态物体；
- parked vehicles 被融合进 static map；
- mapping 使用的帧没有完整 bbox/track annotations；
- 出现 ghost occupancy；
- 相机、LiDAR、bbox 的时间同步或标定存在偏差。

如果同一车辆同时是 Scene GT 和 Entity GT，那么训练会出现矛盾：

```text
Scene loss：这里应该 occupied
Entity loss：这里应该 occupied
Overlap loss：这里不能同时 occupied
```

因此，正式训练前至少需要生成一个用于 Scene Branch 的 cleaned target：

$$
Y_S
=
Y_{\mathrm{map}}
\setminus
\operatorname{CarveOut}(\{B_k\}).
$$

对无法可靠清理的位置使用 ignore，而不是强制赋为 free。

---

## 12. 推荐网络结构

### 12.1 Shared encoders

```text
Multi-view Camera                    LiDAR
        │                              │
        ▼                              ▼
Pretrained Visual Encoder         LiDAR Encoder
        │                              │
        └──── camera-led fusion ◄──────┘
                        │
                        ▼
                 Shared 3D/BEV features
```

Camera 是 object/entity 发起者；LiDAR 主要为视觉 query 提供 metric geometry。

### 12.2 Scene Branch

输入 shared BEV/3D features，输出：

$$
p_S(x),
\qquad
p_S(c\mid x).
$$

负责：

- static occupancy；
- road、sidewalk、wall、building、vegetation 等 scene/stuff semantics；
- bbox 之外的场景结构。

### 12.3 Entity Branch

使用 DETR/MaskFormer-style queries，但第一版只预测 bbox attributes：

$$
q_k
\rightarrow
(\hat B_k,\hat q_k^{\mathrm{entity}},\hat c_k,
\hat s_k^{\mathrm{OOD}},\hat f_k).
$$

视觉 query cross-attend LiDAR features：

$$
Q=F_{\mathrm{entity}}^{\mathrm{camera}},
\qquad K,V=F^{\mathrm{LiDAR}},
$$

$$
F_{\mathrm{entity}}^{\mathrm{fused}}
=
F_{\mathrm{entity}}^{\mathrm{camera}}
+
\operatorname{CrossAttn}(Q,K,V).
$$

职责分配：

| 预测内容 | 主要信息来源 |
|---|---|
| 是否存在独立 entity | Camera |
| objectness / appearance | Camera |
| known semantic / open vocabulary | Camera |
| OOD score | Camera 为主，融合几何冲突 |
| metric center | LiDAR 为主 |
| bbox size / orientation | Camera + LiDAR |
| box occupancy | bbox rasterizer |

### 12.4 Ownership Composer

接收：

- Scene occupancy；
- rasterized Entity occupancy；
- Entity confidence；
- 可选 ownership logits。

输出：

- final occupancy；
- semantic label；
- instance/entity ID；
- OOD probability；
- scene/entity ownership。

---

## 13. 推荐损失函数

总体训练目标：

$$
\begin{aligned}
\mathcal L={}&
\lambda_S\mathcal L_{\mathrm{scene}}
+\lambda_{\mathrm{det}}\mathcal L_{\mathrm{det}}\\
&+\lambda_E\mathcal L_{\mathrm{entity-occ}}
+\lambda_O\mathcal L_{\mathrm{ownership}}\\
&+\lambda_X\mathcal L_{\mathrm{overlap}}
+\lambda_F\mathcal L_{\mathrm{final-occ}}\\
&+\lambda_{\mathrm{sem}}\mathcal L_{\mathrm{semantic}}
+\lambda_{\mathrm{temp}}\mathcal L_{\mathrm{temporal}}.
\end{aligned}
$$

各项含义：

| Loss | 作用 |
|---|---|
| $\mathcal L_{\mathrm{scene}}$ | 监督 bbox 外 static occupancy |
| $\mathcal L_{\mathrm{det}}$ | objectness、bbox、known class |
| $\mathcal L_{\mathrm{entity-occ}}$ | rasterized box 与 filled-box GT 对齐 |
| $\mathcal L_{\mathrm{ownership}}$ | 区分 free、scene、entity |
| $\mathcal L_{\mathrm{overlap}}$ | 抑制两个 branch 重复 claim |
| $\mathcal L_{\mathrm{final-occ}}$ | 保证组合后的总体 occupancy 正确 |
| $\mathcal L_{\mathrm{semantic}}$ | scene semantic 与 entity semantic |
| $\mathcal L_{\mathrm{temporal}}$ | entity identity、位置和属性的时序一致性 |

### 13.1 推荐训练阶段

为了防止一个分支吞掉另一个分支，可以分阶段训练：

#### Stage A：分别建立两个分支

- 使用 cleaned static map 训练 Scene Branch；
- 使用 bbox/objectness 训练 Entity Branch；
- Entity occupancy 由 GT/predicted bbox rasterization 监督。

#### Stage B：联合训练 Composer

- 加入 ownership；
- 加入 overlap penalty；
- 加入 final occupancy loss；
- 先冻结或降低两个 branch encoder 的学习率。

#### Stage C：OOD 与 temporal

- 加入 entity-level OOD head；
- 加入 class-agnostic/pseudo object proposals；
- 加入 track ID 或 query propagation；
- 研究 static-scene residual 对未知物体发现的帮助。

---

## 14. Static Scene Residual 可以成为 OOD Entity Proposal

这套分解还有一个非常有价值的推论。

Scene Branch 表示模型认为“这个地方正常情况下应该有什么”，当前 camera/LiDAR 表示“现在实际看到了什么”。二者之间的新占据残差可能对应临时物体或 OOD object：

$$
r_t(x)
=
p_{\mathrm{current\ evidence}}(x)
\left(1-p_S(x)\right).
$$

```text
Predicted static scene            Current sensor evidence
         │                                 │
         └───────────── difference ────────┘
                              │
                              ▼
                    unexplained occupancy residual
                              │
                              ▼
                  new/temporary/OOD entity proposal
```

这条 residual proposal 对没有已知类别语义的 OOD object 特别重要：

- 它不要求物体属于已知类别；
- 只要当前观测与 static scene 不一致，就能产生 entity candidate；
- camera 可以进一步判断候选区域是否具有 object boundary；
- LiDAR 可以验证它是否具有真实三维占据和公制位置。

但 residual 也可能来自：

- Scene Branch 漏检；
- 地图变化；
- 标定误差；
- LiDAR noise；
- 天气和反射异常。

所以 residual 应当作为 entity proposal signal，而不是直接作为 OOD label。

---

## 15. 第一版 Baseline：最小可行实现

第一版不引入 SAM 3D，不预测精细 Gaussian，也不试图恢复 object 内部结构。

### 15.1 模型

1. Camera backbone；
2. LiDAR backbone；
3. camera-led feature fusion；
4. Scene occupancy decoder；
5. class-agnostic query-based 3D bbox detector；
6. soft/hard box rasterizer；
7. confidence-gated ownership composer；
8. entity-level semantic/OOD heads。

### 15.2 标签

1. static map；
2. oriented 3D bboxes；
3. bbox-derived entity voxel mask；
4. bbox-carved scene target；
5. free/scene/entity/ignore ownership map；
6. final union occupancy。

### 15.3 必做消融

| 实验 | 目的 |
|---|---|
| 单一 occupancy head | 判断双分支是否必要 |
| 双分支，仅 final loss | 验证是否发生职责塌缩 |
| 分支独立监督 | 验证显式 target 的作用 |
| + overlap loss | 检验重复预测是否减少 |
| + confidence composer | 检验误检擦除问题 |
| class-coupled objectness | OOD 对照组 |
| class-agnostic objectness | 验证未知 entity recall |
| + static residual proposals | 检验 OOD entity 发现能力 |

### 15.4 建议指标

除了最终 occupancy IoU/mIoU，还应单独报告：

- Scene occupancy IoU；
- Entity filled-box occupancy IoU；
- 3D bbox recall / mAP；
- class-agnostic entity recall；
- Scene→Entity leakage；
- Entity→Scene leakage；
- branch overlap ratio；
- known semantic mIoU；
- OOD AUROC / AUPR / FPR95；
- false-positive entity 导致的 Scene deletion rate。

定义 Scene→Entity leakage：

$$
\operatorname{Leak}_{S\rightarrow E}
=
\frac{
\sum_x p_E(x)\mathbf 1[z(x)=\mathrm{scene}]
}{
\sum_x\mathbf 1[z(x)=\mathrm{scene}]
}.
$$

定义 Entity→Scene leakage：

$$
\operatorname{Leak}_{E\rightarrow S}
=
\frac{
\sum_x p_S(x)\mathbf 1[z(x)=\mathrm{entity}]
}{
\sum_x\mathbf 1[z(x)=\mathrm{entity}]
}.
$$

---

## 16. 后续怎样接入 SAM 3D 或 Gaussian？

第一版 Entity Branch 的几何表示是：

$$
\text{entity query}
\rightarrow
\text{3D bbox}
\rightarrow
\text{filled-box occupancy}.
$$

建立 baseline 后，可以逐步升级为：

$$
\text{entity query}
\rightarrow
\text{3D bbox + canonical shape latent}
\rightarrow
\text{compact child Gaussian bundle}.
$$

SAM 3D 的作用不是替换 bbox detector，而是给 canonical shape 或 child Gaussian decoder 提供 object-level 3D prior。

推荐升级顺序：

```text
Phase 1
Entity query → bbox → filled occupancy

Phase 2
Entity query → bbox + coarse shape occupancy

Phase 3
Entity query → bbox/layout + compact Gaussian shape

Phase 4
SAM 3D teacher distillation + observed/completed occupancy separation
```

这样每一步都有独立收益和清晰监督，不会在一开始就把 scene/entity decomposition、OOD、Gaussian shape 和大模型蒸馏全部耦合到一起。

---

## 17. 最终建议

这套数据最合理的第一阶段方案是：

> **让 Scene Branch 预测 bbox 之外的 static occupancy，让视觉主导的 Entity Queries 预测 class-agnostic objectness 和 3D bbox，再通过可微 box rasterizer生成 Entity occupancy。训练时使用 bbox 显式构造 scene/entity ownership；推理时使用置信度控制的 Entity 优先合成，避免 Scene 重复预测物体，也避免错误 bbox 无条件擦除真实场景。**

最关键的工程和建模步骤是：

1. 审计 static map 与 bbox 的重叠和 ghost occupancy；
2. 从 bbox 构造 entity mask、cleaned scene target 和 ownership target；
3. objectness 与 known-class probability 解耦；
4. 对 Scene 和 Entity 分别监督，不能只训练 final union；
5. 使用 confidence-gated composer，不做无条件 hard suppression；
6. 第一版保留 filled-box representation，暂不增加没有 GT 支持的精细 shape decoder；
7. 后续再用 SAM 3D prior 将 bbox occupancy 升级成 compact Gaussian object occupancy。

这一版既充分利用了现有数据的标注结构，也能形成一个清晰的科研问题：

> **显式的 Scene–Entity occupancy decomposition，能否将静态场景重建、实例级物体发现和 entity-level OOD detection 统一到一个 camera–LiDAR occupancy framework 中？**

