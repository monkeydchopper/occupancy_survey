# BEV 开放世界 Occupancy：模型建模说明

> 本文只讨论“世界应该怎样被建模”，暂不讨论网络如何训练、损失函数如何设置、推理阈值如何选择。

配套交互图：[NEO-Occ Modeling Artifact](NEO_OCC_MODELING_ARTIFACT.html)

---

## 0. 一句话定义

我们希望模型表达的不是一个扁平的 BEV semantic label map，而是：

> **一个由支撑面、分高度的阻挡占据场、观测证据、残余结构和一组物体实体共同组成的 BEV 世界；known/unknown 是知识库能否解释某个实体的结果，而不是一种物理占据类别。**

最核心的依赖关系是：

```text
传感器观测
    ↓
物理支撑面与阻挡占据
    ↓
把 occupied world 分解成 entities 和 residual field
    ↓
得到每个 entity 的 object-level representation
    ↓
判断当前知识库能否解释这个 entity
```

即：

```text
occupancy → entity → known / new
```

而不是：

```text
known classification → object existence → occupancy
```

后一种建模会导致一个危险问题：如果陌生物体不符合任何已知类别，它可能连“存在”都不会被模型表达。

---

## 1. 这套思路与 2D OOD detection 的关系

这套建模确实借鉴了 2D open-world object detection 和 open-set panoptic segmentation 中的一个重要思想：

> **先形成完整对象，再判断它是否属于已知类别。**

但它不是简单复制下面这条 2D pipeline：

```text
image feature → object proposal → known classification → reject unknown
```

我们只借鉴“object-level reasoning”，没有把 proposal 当作物理世界的起点。

### 1.1 借鉴的部分

- objectness 与 known-class classification 分离；
- 使用完整 object mask/entity representation，而不是逐像素或逐 cell 判断 OOD；
- entity proposal 是网络提出的候选假设；
- unknown 应当在 object/entity 层聚合证据。

### 1.2 不能直接照搬的部分

| 2D detection | BEV occupancy |
|---|---|
| image background 只是视觉背景 | free 是明确的物理空间状态 |
| 通常只表示当前可见表面 | 还要表示遮挡、未观测和预测占据 |
| proposal 漏掉后对象通常成为 background | proposal 漏掉后阻挡 occupancy 仍必须存在 |
| 一个像素通常只对应最前方表面 | 同一个 BEV 坐标可能有多个高度层 |
| box/mask 表示图像区域 | mask 表示物理 footprint 或占据体积 |
| 目标主要是识别 | 还要支持碰撞判断与路径规划 |

因此，这套模型更准确的来源是四种思想的结合：

1. **2D open-world detection**：objectness 与 known classification 分离；
2. **mask/panoptic segmentation**：完整 entity mask；
3. **3D panoptic scene completion**：stuff/scene structure 与 object entities 共同组成世界；
4. **occupancy mapping**：free、occupied、observed、unobserved 是不同维度。

---

## 2. 为什么不能使用一个大 softmax

一个常见但错误的定义是：

```text
free / car / pedestrian / unknown / unobserved
```

这些标签实际上回答的是不同问题：

| 概念 | 回答的问题 |
|---|---|
| free / occupied | 世界中是否存在物质或阻挡体？ |
| observed / unobserved | 传感器是否提供了直接证据？ |
| car / pedestrian | 已知语义类别是什么？ |
| known / unknown | 当前知识库能否解释这个对象？ |
| object / residual field | occupied cells 是否需要组成独立实体？ |

它们不是同一个随机变量的互斥取值。

例如，一个 voxel/cell 可以同时是：

```text
occupied = true
observed = false
entity = q17
concept = unknown
```

含义是：模型根据上下文认为这里属于一个陌生物体，但当前帧没有直接观测证据。

---

## 3. 第一层：支撑面模型

对每个 BEV cell (u=(x,y))，首先定义支撑面变量：

$$
G_u = \left(H_u^{\text{ground}}, D_u\right).
$$

其中：

- (H_u^{\text{ground}})：地面或支撑面的高度；
- (D_u)：road、sidewalk、terrain、drivable 等支撑属性。

### 为什么必须单独建模支撑面

道路表面在 3D 几何中是一个真实 surface，但对于车辆规划来说，它不是 blocking obstacle。

同一个 BEV cell 可以同时包含：

```text
support surface = road
blocking object = sofa
```

如果把道路和沙发放进一个互斥 semantic label 中，就无法同时表达“这是道路”与“道路上有障碍物”。

因此，在道路场景中应当区分：

```text
support / ground
    与
blocking volume
```

---

## 4. 第二层：分高度的阻挡占据场

对每个 BEV cell (u) 和高度层 (h)，定义：

$$
O_{uh}\in\{0,1\}.
$$

它回答：

> 这个位置、这个高度范围内是否存在会阻挡车辆或机器人的物质？

一个干净的 BEV 表示不应只有一个二维 occupancy bit，而应接近 2.5D：

```text
ground/support layer
low obstacle layer       0–0.5 m
vehicle envelope layer   0.5–2.0 m
high structure layer     2.0 m+
```

例如，同一个位置可能是：

```text
ground: road
0–0.5 m: occupied by tire
0.5–2.0 m: free
2.0 m+: occupied by tree canopy
```

### 为什么不能直接沿高度取 max

如果使用：

$$
O_{\text{BEV}}(x,y)=\max_z O(x,y,z),
$$

那么道路表面、树冠、桥梁和车辆会全部被压成同一种二维 occupied，无法区分是否真正阻挡规划空间。

因此，BEV occupancy 必须保留与规划高度相关的信息。

---

## 5. 第三层：观测证据

对每个 (u,h)，定义观测证据：

$$
R_{uh}=\left(n_{uh}^{\text{hit}},n_{uh}^{\text{pass}}\right),
$$

或者使用等价的 evidence representation。

其中：

- `hit`：LiDAR return 落在该位置；
- `pass`：LiDAR ray 穿过该位置；
- 两者都没有：没有直接观测；
- 两者同时较多：可能存在动态物体、配准误差或建图冲突。

关键点是：

> **Unobserved 不是第三种几何状态，而是证据状态。**

因此可以出现：

| Occupancy | Evidence | 含义 |
|---|---|---|
| occupied high | evidence high | 直接看到了阻挡体 |
| occupied high | evidence low | 根据上下文预测存在被遮挡物体 |
| free high | evidence high | ray 明确穿过，是 confirmed free |
| uncertain | evidence zero | 没有足够信息判断世界状态 |

在 BEV 中，ray evidence 应优先在 3D/高度层内计算，再映射到 BEV；不能简单在二维平面上 ray-cast。

---

## 6. 第四层：将 occupied world 分解成 Entity 与 Residual Field

定义 ownership 变量：

$$
A_{uh}\in\left\{0,\text{residual},1,\ldots,Q\right\}.
$$

其含义为：

- (A_{uh}=0)：free；
- (A_{uh}=q)：该 occupied cell 属于第 (q) 个 entity；
- (A_{uh}=\text{residual})：occupied，但不需要独立实例身份。

阻挡占据由 ownership 派生：

$$
O_{uh}=\mathbb 1[A_{uh}\neq 0].
$$

第 (q) 个 entity mask 为：

$$
M_{q,uh}=\mathbb 1[A_{uh}=q].
$$

因此，整个阻挡世界可以写成：

```text
blocking occupancy
    = entity 1 mask
    ∪ entity 2 mask
    ∪ ...
    ∪ residual occupied field
```

### 6.1 为什么使用 ownership partition

另一种直觉写法是 noisy-OR：

$$
P(O=1)=1-(1-P(S))\prod_q(1-P(B_q)P(M_q)).
$$

它能表达“多个成分共同产生 occupancy”，但存在不可辨识问题：

- 两个重复 queries 可以同时解释同一个对象；
- residual field 和 entity 可以同时解释同一个 cell；
- 仅观察最终 occupancy，无法知道 decomposition 是否正确。

ownership partition 更明确：在完整 3D/分高度表示中，每个 occupied cell 只有一个 owner。

---

## 7. 什么是 Entity

第 (q) 个 entity 可以写成：

$$
E_q=\left(B_q,M_q,z_q,\xi_q\right),
$$

其中：

- (B_q)：entity 是否存在；
- (M_q)：BEV footprint 或分高度 mask；
- (z_q)：object-level semantic embedding；
- (ξ_q)：位置、尺寸、高度、朝向、运动等物理属性。

Entity 的定义是：

> **空间上有界、内部连贯、需要作为一个整体被规划、跟踪或解释的 occupied component。**

注意：entity 的定义中还没有 car、pedestrian 或 unknown。

一个陌生沙发不需要先像任何已知类别，才有资格成为 entity。

### 7.1 Entity 与 Residual Field 的边界

这个区分是任务相关的，而不是哲学上的“什么才算物体”：

| 场景成分 | 推荐表示 |
|---|---|
| 正常连续墙面 | residual occupied field |
| 路边连续植被 | residual occupied field |
| 普通道路表面 | support surface |
| 道路上的沙发 | entity |
| 散落纸箱 | entity |
| 倒在路上的树干 | entity |
| 大面积建筑结构 | residual occupied field |

一个实用定义是：

- 有限范围、需要整体避障或跟踪：entity；
- 连续结构、不需要独立身份：residual field；
- 是否位于道路上：entity 与 support surface 的空间关系。

---

## 8. Entity 与 Object Proposal 的关系

两者非常相似，但不完全相同：

> **Entity 是世界中假设存在的真实物体；proposal/query 是网络对 entity 提出的候选解释。**

```text
latent world entity
        ↓ 网络估计
entity proposal / query
        ↓ ownership 与结构化输出
existence + mask + geometry + embedding
```

传统 object proposal 通常表示：

```text
bounding box + objectness
```

这里的 entity proposal 更接近：

```text
BEV/3D mask
+ existence
+ center / height / size
+ object embedding
```

proposal 与真实 entity 不是一一对应：

- 一个 entity 可能产生多个重复 proposals；
- 一个 proposal 可能错误合并两个实体；
- proposal 可能是 false positive；
- 一个真实 entity 可能没有 proposal。

最重要的区别是：

> **没有 proposal 不等于没有 occupancy。**

在普通 detection 中，proposal 漏检后对象通常消失；在 occupancy world model 中，物理阻挡场仍然存在，只是尚未被成功组织成 entity。

---

## 9. 第五层：Known / New 概念模型

只有 entity 已经成立以后，才定义它的知识变量：

$$
C_q\in\{1,\ldots,K,\text{new}\}.
$$

其中 (K) 是当前部署知识库中的已知概念集合。

从 entity mask 聚合 object embedding：

$$
z_q=\operatorname{Pool}\left(\{F_{uh}:M_{q,uh}=1\}\right).
$$

每个 known class 对应一个支持分布：

$$
p_k(z)=p(z\mid C=k).
$$

这个分布可以由多 prototype、kNN、Gaussian mixture 或 vMF mixture 表示。具体选择属于算法实现，不改变建模本身。

概念后验可以写成：

$$
P(C_q=k\mid z_q,\mathcal K)\propto \pi_k p_k(z_q),
$$

$$
P(C_q=\text{new}\mid z_q,\mathcal K)\propto \pi_0p_0(z_q).
$$

其含义是比较两类解释：

```text
解释 A：这个 entity 来自某个已知类别分布
解释 B：现有知识不足，需要一个新概念
```

因此：

> **Unknown 不是一种物体，而是当前知识库无法解释某个 entity 的事件。**

沙发、轮胎、纸箱和倒下的树干不需要共享一个 unknown prototype。它们共同的只是：无法被当前 known supports 可靠解释。

### 9.1 Unknown 永远相对于 Knowledge Set

如果当前 taxonomy 只有：

```text
car / pedestrian / bicycle
```

那么 sofa 是 new。

如果将来知识库加入 sofa，完全相同的物体就可以成为 known sofa。

所以应明确写成：

$$
P(C_q=\text{new}\mid z_q,\mathcal K),
$$

而不是脱离知识集合地声称某个对象在绝对意义上“是 unknown”。

---

## 10. 完整的模型依赖关系

把世界变量记作：

$$
W=\left\{G,O,R,A,E_{1:Q},C_{1:Q}\right\}.
$$

一个清晰的判别式后验分解是：

$$
q(W\mid X,\mathcal K)
=q(G,O,R\mid X)
\cdot q(A,E_{1:Q}\mid O,G,X)
\cdot\prod_q q(z_q\mid M_q,X)
p(C_q\mid z_q,\mathcal K).
$$

各项分别回答：

1. (q(G,O,R\mid X))：世界中的支撑面、阻挡占据和观测证据是什么？
2. (q(A,E\mid O,G,X))：occupied world 应怎样被组织成 entities 和 residual field？
3. (q(z_q\mid M_q,X))：完整 entity 的 object representation 是什么？
4. (p(C_q\mid z_q,\mathcal K))：当前知识库能否解释它？

这是一种世界变量与依赖关系的定义，不等于必须使用某种特定神经网络：

- entity 可以由 mask queries 实现；
- 也可以由 center-offset grouping 实现；
- 还可以由 connected components、图聚类或 tracking entities 实现。

建模应当独立于具体实现。

---

## 11. “道路上的沙发”完整例子

假设车辆观测到道路上有一个从未见过的沙发。

### 支撑面

```text
ground height = 0.0 m
support type = road
drivable = true（如果没有障碍物）
```

### 阻挡占据

```text
0–0.5 m = occupied
0.5–1.5 m = occupied
1.5 m+ = mostly free
```

### 观测证据

```text
LiDAR hit evidence = high
ray pass evidence before the sofa = high
space behind the sofa = partially unobserved
```

### Entity ownership

```text
sofa occupied cells → entity q17
road surface cells   → support layer
nearby vegetation    → residual field
```

### 知识解释

```text
p_car(z17)        = low
p_pedestrian(z17) = low
p_bicycle(z17)    = low
p_new(z17)        = high
```

因此，沙发同时具有以下状态：

```text
位于 road support 上
+ 是 blocking occupied volume
+ 是一个完整 entity
+ 相对于当前 taxonomy 是 new concept
```

这四个结论属于不同变量，可以同时成立。

---

## 12. 这套模型明确不是什么

### 12.1 不是 K+1 semantic classifier

它不要求所有 unknown objects 形成一个共同的类别中心。

### 12.2 不是普通 detector 加一个 OOD score

物理 occupancy 不依赖 proposal 或 known classification 是否成功。

### 12.3 不是把 unobserved 当作几何类别

Unobserved 描述证据不足，不描述真实世界一定 free 或 occupied。

### 12.4 不是纯二维单层 BEV

它需要保留支撑面和阻挡高度，否则无法区分 road、低矮障碍、车辆、树冠和桥梁。

### 12.5 不是所有 unlabeled occupied 都算 unknown object

未标注建筑、墙体和植被可以进入 residual field；只有形成独立 entity 后，才讨论 known/new。

---

## 13. 最终模型摘要

整套建模可以压缩成五句话：

1. **支撑面与阻挡占据分开。** 道路是 support，道路上的沙发是 blocking entity。
2. **物理状态与观测证据分开。** Unobserved 不是 free，也不是 unknown。
3. **Occupied world 被分解为 entities 与 residual field。** 不是所有 occupied structures 都需要实例身份。
4. **Entity 是世界变量，proposal 是网络假设。** Proposal 漏检不能让物理 occupancy 消失。
5. **Unknown 是知识事件。** 它表示完整 entity 无法被当前 known concept set 解释。

最终定义为：

> **BEV 世界首先是“支撑面 + 分高度阻挡占据场”；阻挡场再被分解成 residual structures 与一组 entities；known/unknown 是当前知识库对 entity 的解释结果，而不是 occupancy 类型。**

