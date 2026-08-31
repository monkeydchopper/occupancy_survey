# S2GO 讲解：用时序 Query 驱动稀疏 3D Gaussian Occupancy

> 论文：**S2GO: Streaming Sparse Gaussian Occupancy Prediction**（ICLR 2026）  
> 核心定位：以多目相机为输入，用少量可持续传播的 3D query 表示场景；每个 query 再生成一组局部 3D Gaussian，最后把这些 Gaussian 转成稠密 semantic occupancy。  
> 本文先忠实解释 S2GO，再讨论它如何改造成 camera–LiDAR、视觉主导的 OOD occupancy 模型。
>
> 配套交互讲解：[S2GO Query → Gaussian Artifact](./S2GO_QUERY_TO_GAUSSIAN_ARTIFACT.html)

---

## 0. 一句话先说清楚

S2GO 的核心不是“拿 Gaussian 替代 voxel”这么简单，而是：

> **用少量 persistent query 维护场景的时序记忆，每个 query 负责生成一小簇局部 Gaussian 来描述空间形状，再由 Gaussian-to-Voxel 算子恢复稠密 occupancy。**

可以把它理解成三层表示：

```text
场景级时序记忆
      │
      ▼
Persistent Queries             少量、可传播、负责“去哪里看”
      │ 每个 query 解码 J 个 Gaussian
      ▼
Local Gaussian Bundles         连续、稀疏、负责“局部形状是什么”
      │ Gaussian-to-Voxel
      ▼
Dense Semantic Occupancy       规则体素输出，负责最终评测与下游使用
```

它同时想解决三个问题：

1. **稠密 voxel 很贵**：不希望在整个 3D 空间维护密集 feature volume。
2. **单帧相机深度不可靠**：需要时序信息和更强的 3D 几何初始化。
3. **普通 Gaussian 数量很多且缺乏高层组织**：希望用 query 作为父节点，把局部 Gaussian 组织起来。

---

## 1. 先把几个容易混淆的概念分开

### 1.1 Query、Gaussian、Voxel 各自是什么？

| 层级 | 数量 | 表示什么 | 是否最终输出 |
|---|---:|---|---|
| Query | 少 | 一个带 3D 位置和时序状态的父节点 | 否 |
| Child Gaussian | 中等 | 父 query 附近的一块连续 3D 形状 | 否 |
| Voxel | 多 | 规则网格上的 occupancy 与语义 | 是 |

直觉上，一个 query 像一根“会移动、会记忆的图钉”，钉在场景中的某个位置；它周围展开若干个椭球 Gaussian，去覆盖真实表面或物体体积。

```text
                  child Gaussian
                       ◯
                 ◯     │     ◯
                   \   │   /
                    [ Query ]  ── velocity / temporal state
                   /   │   \
                 ◯     │     ◯
                       ◯
```

### 1.2 S2GO 的 query 是不是 object instance？

**不是严格意义上的 instance。** 更准确地说，它是一个 persistent spatial-semantic anchor。

它有一点“object-like”：

- 一个 query 的 child Gaussians 共享父节点的语义和速度；
- query 可以随时间传播；
- 局部 Gaussian bundle 容易形成连贯结构，所以可视化上更像完整物体。

但论文没有强制：

- one query = one object；
- query 与 instance GT 一一匹配；
- 使用 instance mask、track ID 或 Hungarian instance matching；
- 每个物体只能由一个 query 表示。

因此，一个 query 可能表示：

- 一辆车的大部分；
- 半辆车；
- 一块道路；
- 一段墙；
- 两个距离很近、同语义、运动相似的小结构。

所以我们不能直接说“**S2GO 已经有 entity 层**”。更合适的说法是：

> S2GO 已经提供了一个很好的 **entity precursor（实体原型）**，但若用于 OOD object，需要再加 objectness、instance consistency 和 unknown-aware semantics，才能变成真正的 entity。

### 1.3 它和 MaskFormer / PaSCo 的关系

三者都有“少量 query 组织稠密输出”的共同思想，但监督目标不同：

| 方法 | Query 最终负责什么 | Query 是否被显式约束为实例 |
|---|---|---|
| MaskFormer 系列 | 2D mask + class | 通常是 segment / instance 级 |
| PaSCo | 3D panoptic mask + class | 是，面向 3D instance / panoptic |
| S2GO | 一组局部 Gaussian + 共享语义/运动 | 否，主要是 occupancy 表示单元 |

S2GO 更接近“**query-conditioned Gaussian field**”，而不是直接输出 instance mask 的 MaskFormer。

---

## 2. 整体网络结构

### 2.1 原论文的输入输出

- **Stage 1 预训练**：多目 RGB；LiDAR 仅用于 query 几何去噪目标；邻帧 RGB/depth 用于可微渲染监督。
- **Stage 2 occupancy 训练与推理**：多目 RGB；最终输出 semantic occupancy。
- 原版 S2GO **不是 camera–LiDAR 在线融合模型**。LiDAR 在原论文中主要是训练阶段的几何老师。

### 2.2 端到端数据流

```text
过去若干帧保存的 queries
          │
          │  Temporal Attention
          ▼
当前 learnable queries  ◄────────────────────────────┐
          │                                           │
          │ Deformable Cross-Attention                │
          ▼                                           │
多目图像 ──► ResNet-50 + FPN ──► multi-scale features │
          │                                           │
          ▼                                           │
更新后的 query state: (feature, 3D position)           │
          │                                           │
          ├── parent offset / opacity / velocity      │
          │                                           │
          └── J 个 child Gaussians                    │
                    │                                 │
                    ├── mean offset                   │
                    ├── rotation                      │
                    ├── scale                         │
                    ├── opacity                       │
                    └── semantic feature              │
                              │                       │
                              ▼                       │
                    Gaussian-to-Voxel                 │
                              │                       │
                              ▼                       │
                    dense semantic occupancy          │
                                                      │
挑选高价值且空间分散的 queries ───────────────────────┘
                    传播到下一帧
```

论文中的 Temporal Transformer 沿用 PETR / StreamPETR 一类思路：当前 query 一方面从相机特征取信息，另一方面与历史 query 状态交互。历史缓存覆盖约 4 帧、2 秒；特征维度为 768，并使用 FlashAttention 提高效率。

### 2.3 为什么 query 比直接传播 Gaussian 更合理？

假设一帧有 900 个 query、每个 query 产生 10 个 Gaussian：

- 传播所有 Gaussian：要传播 9000 个低层 primitive；
- 传播 query：只需传播 900 个高层状态，再重新解码局部形状。

query 压缩了以下信息：

```text
位置 + 外观/语义 feature + 局部形状摘要 + 运动趋势
```

因此 temporal module 不必追踪每一个椭球的细节，只需追踪父节点。

### 2.4 Past / New / Future 3D Queries 到底是什么？

Figure 1 中的三个名字很容易让人以为模型分别预测过去、现在和未来三套 query。仔细对照正文后，更准确的理解是：

> **Past、New、Future 是同一种 3D query 在当前时间步中的三个角色，不是三种不同结构，也不是三个独立的预测 head。**

完整循环是：

```text
上一时刻被选中的 Future Queries
                 │
                 ▼
         当前时刻 Past Queries
                 │ ego transform
                 │
New Queries ─────┼──► Temporal Transformer ◄── Current Images
                 │
                 ▼
         Refined Current Queries
          │                  │
          │                  └──► Gaussians → 当前 Occupancy
          │
          └── 选择一部分
                 ▼
           Future Queries
                 │
                 ▼
        下一时刻成为 Past Queries
```

可以用一个公式概括当前时间步：

$$
\mathcal Q_t^{\mathrm{refined}}
=
\operatorname{TemporalTransformer}
\left(
\mathcal Q_t^{\mathrm{new}},
\operatorname{EgoAlign}(\bar{\mathcal Q}_t),
\mathcal F_t^{\mathrm{image}}
\right).
$$

其中 $\bar{\mathcal Q}_t$ 是 past-query queue。

#### 2.4.1 New 3D Queries：当前帧的新工作槽位

“New”不是“新出现物体的 query”，而是：

> 每个时间步新拿出来、等待当前图像和历史信息更新的一组 query slots。

New Query 的 3D location 在两个训练阶段来源不同。

**Stage 1：从带噪 LiDAR 点产生。**

从当前帧点云中做 Farthest Point Sampling：

$$
\hat p_t^i
=
\operatorname{FPS}_K(\mathrm{pts}_t)_i,
$$

然后加入随机噪声：

$$
p_t^{i,\mathrm{new}}
=
\hat p_t^i+\epsilon_i,
\qquad
\epsilon_i\sim\mathcal U(-e,e).
$$

网络预测 query offset，使它从带噪位置回到 LiDAR 表面：

$$
p_t^{i,\mathrm{refined}}
=
p_t^{i,\mathrm{new}}+o_t^i.
$$

```text
真实 LiDAR 表面点 ×
                  ▲
                  │ predicted offset
带噪 New Query   ●
```

因此，在 Stage 1 可以说 **LiDAR 直接指导了 New Query 的空间初始化**。但这里没有 LiDAR encoder，也没有在线 LiDAR feature fusion；点云仅用于初始化 location、构造 denoising target，以及生成渲染所需的 depth supervision。

**Stage 2 和正常推理：来自 learnable 3D locations。**

论文 §3.4.1 明确写道，occupancy 阶段的 query positions 被初始化在可学习的 3D locations：

$$
P^{\mathrm{learn}}
=
\{p_1^{\mathrm{learn}},\ldots,p_K^{\mathrm{learn}}\}.
$$

于是每一帧的 New Query bank 可以概念性地写成：

$$
\mathcal Q_t^{\mathrm{new}}
=
\left\{
(q_i^{\mathrm{init}},p_i^{\mathrm{learn}})
\right\}_{i=1}^{K}.
$$

论文前文使用了“randomly initialized throughout the 3D scene”的表述，后文使用“learnable 3D locations”。二者可以统一理解为：

```text
训练开始：3D location 参数随机初始化
                  ↓
Stage 2：通过 occupancy loss 学习这些 location
                  ↓
正常推理：每帧从学到的 query anchors 开始
```

也就是说，它们不是每帧根据 LiDAR 产生，不是每帧重新随机采样，也不是先从当前图像检测物体再动态创建。

论文没有进一步公开 query content embedding 初始化和具体 tensor 拼接的伪代码；根据其 PETR / StreamPETR 式架构，可以确认的是 current query bank 与 past-query memory、当前图像 feature 进入 Temporal Transformer，但不应在没有代码的情况下虚构更细的实现。

#### 2.4.2 Past 3D Queries：过去输出形成的稀疏记忆

Past Queries 来自之前时间步被保存的 queries：

$$
\bar{\mathcal Q}_t
=
\left\{
\mathcal Q_{t-1}^{\mathrm{future}},
\mathcal Q_{t-2}^{\mathrm{future}},
\ldots
\right\}.
$$

论文实现使用约 4 帧、2 秒的历史 queue。

由于过去 query 的 3D location 位于过去的 ego coordinate system，送入当前 Temporal Transformer 前需要先做 ego-motion 对齐：

$$
p_{\tau\rightarrow t}
=
T_{t\leftarrow\tau}^{\mathrm{ego}}p_\tau.
$$

```text
过去 ego frame 中的 query
             │
             │ Ego Transform
             ▼
当前 ego frame 中的 Past Query
```

Past Query 主要承担 memory 的角色：它把过去已经提取到的位置、语义/外观 feature 和场景上下文提供给当前 New Queries。当前 query 再同时通过：

- temporal/self-attention 与 Past Queries 交互；
- deformable cross-attention 从当前多目图像提取新证据。

需要特别注意：Figure 1 明确画出了 past-query location 的 **Ego Transform**。论文明确说 velocity 用于邻帧 RGBD/occupancy supervision 和 future-occupancy rollout，但没有明确写 past memory 的位置一定通过 $p+v\Delta t$ 外推。因此不能把 Past Query 对齐直接描述成“ego transform + velocity motion compensation”，除非后续代码或作者说明确认这一点。

Past Query 也不等于严格的 track：它没有 instance matching 和 track-ID supervision，本质上仍然是过去保存下来的 sparse scene memory。

#### 2.4.3 Future 3D Queries：准备送往未来的当前结果

Future Queries 不是另一个 decoder 额外生成的 query bank，而是当前 refined queries 中被选中进入历史 queue 的子集：

$$
\mathcal Q_t^{\mathrm{future}}
=
\operatorname{Select}_{\mathrm{opacity},\delta}
\left(
\mathcal Q_t^{\mathrm{refined}}
\right).
$$

作者首先优先选择 query opacity 较高的候选，因为它们更可能对应 occupied regions；然后要求被选 query 两两之间保持最小距离：

$$
\|p_i-p_j\|>\delta.
$$

距离约束避免所有 memory 都聚集在少数高-opacity 区域。训练时 $\delta$ 在 0–3 m 之间随机采样，推理时约为 1.6 m。

下一帧到来时：

$$
\mathcal Q_t^{\mathrm{future}}
\longrightarrow
\mathcal Q_{t+1}^{\mathrm{past}}.
$$

因此 “Future” 的准确含义是：

```text
will be propagated to future timesteps
```

而不是：

```text
已经被预测到未来位置的 query
```

#### 2.4.4 Future Queries 不等于 Future Occupancy

论文 Figure 4 还展示了 future occupancy prediction，这和 Figure 1 的 Future Queries 是两件事。

| 概念 | 实际操作 | 目的 |
|---|---|---|
| Future Query | 选择当前 query，存入 temporal queue | 给后续帧提供 memory |
| Future Occupancy | 根据预测 velocity 移动 query/Gaussian | rollout 未来空间占据 |

Future occupancy 可以近似写成：

$$
\mu_i(t+\Delta t)
=
\mu_i(t)+v_i\Delta t.
$$

所以不能因为图中写了 Future 3D Queries，就认为 transformer 已经预测了一组未来轨迹 query；它主要表示“会被传播到未来的记忆”。

#### 2.4.5 按时间展开看一遍

第一帧没有历史 memory：

```text
Past Queries = empty
New Queries = learnable query bank
                    │
                    ▼
       Image + Temporal Transformer
                    │
                    ▼
          Refined Current Queries
             ├── Gaussians → occupancy
             └── Select → Future Queries
```

下一帧：

```text
上一帧 Future Queries
          │
          ▼
    Ego Transform
          │
          ▼
Past Queries ───────────────┐
                            │
Learnable New Queries ──────┼──► Temporal Transformer
                            │
Current Images ─────────────┘
                            │
                            ▼
                  Refined Current Queries
                    ├── current occupancy
                    └── new Future Queries
```

最终可以记成一句话：

$$
\boxed{
\text{New 是当前工作槽位，Past 是历史记忆，Future 是准备保存的当前结果}
}
$$

这也修正了一个可能的误解：用“当前未解释的 LiDAR evidence”在 Stage 2 动态生成 New Queries，是我们可以提出的 **LiDAR-driven query birth 扩展**，不是原始 S2GO 的做法。原方法只有 Stage 1 的 New Query locations 来自带噪 LiDAR 点；Stage 2 与推理仍使用 learnable 3D locations。

---

## 3. Query 如何生成一组 Gaussian？

设第 $i$ 个 query 的当前位置为 $p^i$，feature 为 $q^i$。网络先预测父节点属性：

- 父 offset：$o^i$
- 父 opacity：$a^i$
- velocity：$v^i$

然后为该 query 解码 $J$ 个 child Gaussian。第 $j$ 个 child 还预测：

- 局部 offset：$o_j^i$
- rotation：$r_j^i$
- scale：$s_j^i$
- child opacity：$a_j^i$

于是整帧 Gaussian 集合可以写成：

$$
\mathcal G_t =
\left\{
\left(
p^i + o^i + o_j^i,
v^i,
r_j^i,
s_j^i,
a^i a_j^i
\right)
\right\}_{i=1,j=1}^{K,J}.
$$

这里最关键的是两级结构：

```text
父 query：决定这组 Gaussian 大致在哪里、是否存在、如何运动
child：   决定父节点附近的细粒度形状、方向和尺度
```

论文配置中：

| 配置 | Query 数 K | 每个 Query 的 child 数 J | Gaussian 总数 |
|---|---:|---:|---:|
| Small | 900 | 10 | 9,000 |
| Base | 1,800 | 20 | 36,000 |

Occupancy 阶段，同一个 query 产生的 child Gaussians 共享语义类别。这让局部结构更连贯，但也意味着它并不适合让同一个 query 横跨语义边界。

---

## 4. 为什么需要两阶段训练？

S2GO 最重要、也最容易被忽略的贡献之一，是 **query pretraining**。

### 4.1 直接做 occupancy 为什么不够？

最初的 learnable query 可能落在任意 3D 位置，很多会落在空空间。如果只用最终 voxel occupancy loss：

- loss 告诉模型“这里预测错了”；
- 但不明确告诉某个 query 应该往哪个方向移动；
- query 离真实表面较远时，局部 Gaussian 对正确 voxel 的梯度可能很弱；
- 最终容易出现 query 聚集、漂移或留在空处。

这和目标检测中 object query 一开始不知道该去哪里类似，但 3D occupancy 的搜索空间更大，监督又更稠密、更局部，优化更难。

S2GO 的做法是：先教 query **找到被占据的空间，并学会用局部 Gaussian 重建那里**，然后再训练完整 semantic occupancy。

---

## 5. Stage 1：Geometry-aware Query Pretraining

### 5.1 用带噪 LiDAR 点教 query“从附近走到表面”

先从当前帧 LiDAR 点云中用 Farthest Point Sampling 选出 $K$ 个目标点：

$$
\hat p_i = \operatorname{FPS}_K(\mathrm{pts}_t)_i.
$$

然后给这些点加入均匀噪声，作为 query 的初始位置：

$$
p_i = \hat p_i + \epsilon_i,
\qquad
\epsilon_i \sim \mathcal U(-e,e).
$$

nuScenes 实验中 $e=1\text{ m}$。网络预测 offset $o_i$，使 query 回到真实表面附近：

$$
\mathcal L_{\mathrm{denoise}}
=
\sum_i \left\|\hat p_i-(p_i+o_i)\right\|.
$$

```text
LiDAR surface point ×
                    ▲
                    │ predicted offset
                    │
noisy query       ●
```

为什么一定要加噪声？如果直接把 query 放在精确 LiDAR 点上，模型只会学会“保持原位”；加入噪声才迫使它学习：

> 根据多目图像线索，从附近的空空间寻找真实 3D 表面。

### 5.2 Render RGB 和 depth，约束局部 Gaussian

query 回到表面还不够，它还要生成正确的局部 Gaussian。论文把预测 Gaussian 投影/渲染到当前帧及相邻关键帧，并监督：

- RGB reconstruction；
- depth reconstruction。

总预训练目标可概括为：

$$
\mathcal L_{\mathrm{pre}}
=
\lambda_{\mathrm{den}}\mathcal L_{\mathrm{denoise}}
+\lambda_{\mathrm{depth}}\mathcal L_{\mathrm{depth}}
+\lambda_{\mathrm{rgb}}\mathcal L_{\mathrm{rgb}}.
$$

渲染相邻时刻时，利用 ego-motion 和 query velocity 把 Gaussian 移到对应时刻。因此 velocity 不是最后才附加的属性，而是在几何预训练阶段就受到多帧重建约束。

### 5.3 这个预训练到底学到了什么？

它不是简单“用 LiDAR 做 occupancy GT”，而是在学三个更基础的能力：

1. **Traversal**：query 如何从空处移动到可观测表面。
2. **Local geometry**：一组 child Gaussian 如何覆盖局部形状。
3. **Motion consistency**：同一局部结构如何在邻帧保持一致。

论文的 ablation 很能说明问题：

| 预训练方式 | mIoU | IoU |
|---|---:|---:|
| 无预训练 | 13.02 | 25.73 |
| 只用精确 LiDAR 点初始化 | 13.62 | 27.08 |
| 加噪 query + RGB/depth render，不做 denoise | 20.55 | 32.68 |
| 完整预训练：render + denoise | **21.60** | **33.91** |

结论不是“LiDAR 点越准越好”，而是：

> **query 必须学会从错误位置主动回到真实几何结构；这个优化能力比把 query 直接放在正确点上更重要。**

### 5.4 没有 LiDAR 能不能预训练？

可以弱化对 LiDAR 的依赖。论文还用 Metric3D 的 zero-shot monocular depth 作为几何初始化，性能从 21.60/33.91 降到 20.99/33.57，差距不大。

不过要注意：

- 这说明 query pretraining 可以用强单目深度模型近似；
- 不说明 monocular depth 与 LiDAR 在所有天气、远距离、小物体上等价；
- 对我们计划做的 camera–LiDAR fusion，保留 LiDAR 几何监督仍然更自然。

---

## 6. Stage 2：Semantic Occupancy Training

Stage 2 中，query 不再从带噪 LiDAR 点启动，而是从可学习的 3D reference location 开始。网络只根据多目 RGB 和历史 query：

1. 更新 query feature 与位置；
2. 解码带语义、opacity、velocity 的 child Gaussians；
3. 用 Gaussian-to-Voxel 得到规则网格；
4. 使用 dense semantic occupancy GT 监督最终输出；
5. 使用相邻时刻 occupancy/motion 信号约束时序行为。

因此原始 S2GO 在正常推理时仍然是 camera-only：

```text
LiDAR：Stage 1 的训练老师
Camera：Stage 1/2 的主要观测，也是最终推理输入
```

这一点对我们很重要：如果要在推理时使用 LiDAR 做几何校正，需要真正改网络，而不能只说“沿用 S2GO”。

---

## 7. Gaussian-to-Voxel：连续 Gaussian 如何变回 occupancy？

### 7.1 单个 Gaussian 对空间点的影响

对第 $i$ 个 Gaussian，中心为 $\mu_i$，协方差为 $\Sigma_i$。它对 voxel center $x$ 的空间响应为：

$$
g_i(x)=
\exp\left[
-\frac{1}{2}(x-\mu_i)^\top
\Sigma_i^{-1}(x-\mu_i)
\right].
$$

加入预测 opacity $a_i$ 后：

$$
\alpha_i(x)=a_i g_i(x).
$$

### 7.2 多个 Gaussian 如何合成 occupied probability？

把每个 Gaussian 看成一次独立的“占据机会”，空间点 $x$ 的空闲概率是所有 Gaussian 都没有占据它：

$$
P_{\mathrm{free}}(x)
=
\prod_i (1-\alpha_i(x)).
$$

于是 occupied probability 为：

$$
P_{\mathrm{occ}}(x)
=
1-\prod_i(1-\alpha_i(x)).
$$

这比简单把 Gaussian response 相加更合理，因为结果天然在 $[0,1]$ 内，也表达了多个 Gaussian 的联合支持。

### 7.3 语义如何合成？

设第 $i$ 个 Gaussian 的语义分布或语义 feature 为 $\tilde c_i$，则 occupied 条件下的语义可以用空间响应与 opacity 加权：

$$
e(x)=
\frac{
\sum_i p(x\mid\mathcal G_i)a_i\tilde c_i
}{
\sum_j p(x\mid\mathcal G_j)a_j
}.
$$

最终一个 voxel 的输出可写成：

$$
P(x)=
\left[
P_{\mathrm{occ}}(x)e(x),
1-P_{\mathrm{occ}}(x)
\right],
$$

即“各 occupied semantic class + free”。

### 7.4 为什么 opacity 很关键？

如果 G2V 只看 Gaussian 的中心和尺度，不看 opacity，渲染分支和 occupancy 分支会出现语义不一致：

- Gaussian 可以把 opacity 降得很小，在图像渲染中近似消失；
- 但它仍可能在 voxelization 时被视为实体；
- 模型甚至可能通过缩小、漂到 voxel 间隙等方式钻监督漏洞。

S2GO 把 opacity 同时用于 rendering 和 G2V，使“这个 Gaussian 是否真实存在”在两个分支中统一。论文 ablation 中，opacity-aware G2V 带来约 **+3.16 mIoU**。

### 7.5 计算量怎么控制？

朴素实现需要对“所有 voxel × 所有 Gaussian”逐对计算。S2GO 的 CUDA G2V 算子把 voxel 划成 $4\times4\times4$ block，只计算会影响该 block 的 Gaussian。

在论文给出的 9k Gaussians、约 640k voxels 设置中，相对基线实现报告：

- forward 约 1.5× 加速；
- backward 约 20.4× 加速。

所以 Gaussian 稀疏并不自动等于高效，**高效的 G2V 实现本身也是这套方法成立的关键**。

---

## 8. Temporal Query Propagation：时序是怎么做的？

### 8.1 传播什么？

S2GO 不传播完整 voxel grid，也不传播所有 child Gaussians，而是传播一部分父 query：

```text
第 t 帧：image → update queries → decode Gaussians → occupancy
                         │
                         └── select persistent queries
                                      │
                                      ▼
第 t+1 帧：new queries + propagated queries → temporal attention
```

这样历史长度增加时，不必把所有历史 voxel/Gaussian 重算一遍；时序状态保持为固定规模的 query 表示。

### 8.2 哪些 query 会留下？

作者使用两条规则：

1. **Top-k opacity**：优先保留模型认为真实存在的 query。
2. **Minimum distance filtering**：已选 query 附近一定距离内，不再选重复 query，保证空间多样性。

训练时距离阈值 $\delta$ 在 0–3 m 随机采样；推理时使用约 1.6 m。

```text
候选 query：   ● ●●      ●       ●●

只按 opacity： ● ●       ●       ●
               可能仍集中

加距离约束：   ●         ●       ●
               覆盖更分散
```

Ablation：

| 传播策略 | mIoU | IoU |
|---|---:|---:|
| 不传播 | 17.92 | 29.24 |
| Top-k opacity | 19.94 | 32.03 |
| Top-k + distance filtering | **20.51** | **32.51** |

### 8.3 Velocity 是怎么用的？

每个 query 预测一个 velocity，主要有两种用途：

- 把 query / Gaussian 对齐到相邻时刻，提供时序重建和监督；
- 将当前 Gaussian 向未来 rollout，产生短期 future occupancy。

但这不是一个完整的行为预测系统。它基本假设局部 query 按预测速度传播，没有显式建模物体交互、地图约束或多模态未来。

---

## 9. 它为什么比“每个 Gaussian 独立预测”更有意思？

### 9.1 高层组织约束

如果每个 Gaussian 都独立预测语义、运动和位置，局部形状容易碎裂：

```text
独立 Gaussians：     ◯    ◯  ◯       ◯
                       ◯       ◯

Query + children：       [Q]
                       /  |  \
                      ◯   ◯   ◯
                       \  |  /
                         ◯
```

父 query 让一组局部 primitive 共享语义、存在性和运动趋势，形成一种低成本结构先验。

### 9.2 时序状态更紧凑

query 负责“记忆”，child Gaussian 负责“展开”。这是一种很实用的 temporal compression：

$$
\text{compact state}
\xrightarrow{\text{decoder}}
\text{detailed geometry}.
$$

### 9.3 兼顾连续几何与标准输出

- Gaussian 对连续表面、尺度和方向表达自然；
- voxel 输出仍可直接使用主流 occupancy GT、指标与规划接口；
- rendering 分支又能接 RGB、depth 或 foundation-model feature supervision。

这是它对我们最有价值的地方：**中间表示连续，监督和输出不必被 Gaussian 绑定。**

---

## 10. S2GO 的强项与局限

### 10.1 强项

1. **表示层次清楚**：query 管高层状态，Gaussian 管局部形状，voxel 管最终输出。
2. **时序高效**：传播少量 query，而非巨大 BEV/voxel feature。
3. **优化设计扎实**：geometry-aware denoising 解决 query 不会寻找表面的难题。
4. **rendering 与 occupancy 一致**：opacity-aware G2V 把两个监督空间连起来。
5. **自然支持运动**：query velocity 可用于对齐和短期 rollout。
6. **速度较好**：论文报告 Small 模型在 A100 上约 20.8 FPS，Base 约 14.5 FPS；具体数值不能脱离其分辨率、数据处理和硬件设置横向比较。

### 10.2 对普通 occupancy 的局限

1. 一个 query 的 children 共享语义和速度，跨物体或跨语义边界时表达会受限。
2. query 不是真 instance，不能直接给稳定 object ID 或 panoptic mask。
3. 两阶段训练中 query 初始化方式存在 domain gap：预训练从带噪表面附近出发，occupancy 阶段从 learnable locations 出发。
4. future rollout 更像 constant-velocity extrapolation，而非强交互预测。
5. 训练依赖多种 supervision 和专用 G2V 实现，复现成本不低。

### 10.3 对 OOD object 尤其危险的地方

#### 局限 A：closed-set semantic head 会把 unknown 压进 known class

原始模型最终仍做固定类别 semantic occupancy。遇到从未见过的物体，feature 会被迫选择某个已知类别。

#### 局限 B：opacity 并不等于 OOD objectness

高 opacity 只说明某个 Gaussian/区域对 occupancy 有贡献，不说明它是一个独立物体，更不说明它是未知物体。

#### 局限 C：Top-k propagation 容易删掉最重要的 OOD

新出现、远距离、小尺寸、遮挡严重的 OOD 往往：

- opacity 低；
- 单帧不稳定；
- 恰恰需要时序积累才能被确认。

只按 opacity 保留 query，会形成一个坏循环：

```text
初始证据弱 → 不被传播 → 无法累积证据 → 一直证据弱
```

#### 局限 D：distance filtering 可能压掉相邻小物体

若两个小 OOD 距离很近，固定的 1.6 m suppression 可能只保留一个。对行人附近散落物、施工设施、动物等尤其敏感。

#### 局限 E：没有明确区分 stuff 与 entity

道路、墙、植被等大面积 stuff 也会占用 query。若所有 query 统一竞争传播预算，小 OOD entity 很容易输给高面积、高 opacity 的背景结构。

---

## 11. 对我们的 OOD Occupancy，S2GO 应该怎么用？

最稳妥的定位是：

> **把 S2GO 当作 sparse temporal occupancy backbone，而不是把它直接当作 OOD detector。**

它提供表示和时序骨架；OOD 需要重新设计 entity、语义空间和传播规则。

### 11.1 概念映射

| S2GO 原组件 | 在我们的模型中保留什么 | 需要新增什么 |
|---|---|---|
| Persistent query | 稀疏时序状态 | entity existence、identity consistency |
| Parent opacity | 占据存在证据 | 与 objectness / OOD 分数解耦 |
| Shared semantic class | 局部语义一致性 | open-vocabulary feature、known/new posterior |
| Query velocity | 时序运动先验 | uncertainty、遮挡下 persistence |
| Child Gaussians | entity / stuff 的局部 3D 形状 | 多尺度 children、边界约束 |
| G2V | 输出标准 BEV/3D occupancy | known / unknown / free 分解 |
| RGB-depth rendering | 2D–3D 几何监督 | VLM feature、mask、ray evidence rendering |
| Top-k propagation | 固定预算时序记忆 | risk-aware、OOD-aware retention |

### 11.2 我建议的总体结构

```text
                         ┌──────────────────────────┐
多目 Camera ─► VLM/DINO │ multi-view visual feature│
                         └─────────────┬────────────┘
                                       │ 视觉主 cross-attention
历史 entity/stuff queries ─────────────┼──────────────┐
                                       ▼              │
                               Query Transformer      │ temporal attention
                                       │              │
LiDAR ─► sparse voxel encoder ─────────┤ 几何 cross-attention
                                       │              │
                    ┌──────────────────┴──────────────┘
                    ▼
       updated query = geometry + semantics + memory
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  parent/entity heads       child Gaussian decoder
  - existence              - mean / scale / rotation
  - objectness             - opacity
  - velocity               - open semantic feature
  - known/new score                │
  - uncertainty                    ▼
          │                 opacity-aware G2V
          │                        │
          └──────────────┬─────────┘
                         ▼
              known / unknown / free occupancy
```

这里的主从关系应该很明确：

- **Camera 决定“这是什么、是否语义异常”**；
- **LiDAR 决定“它准确在哪里、几何上是否被观测到”**；
- **Temporal query 决定“它是否持续存在、单帧弱证据是否应该积累”**。

### 11.3 Camera 主导、LiDAR 做几何校正

可以用两步非对称 cross-attention：

$$
q_i^{\mathrm{vis}}
=
q_i+operatorname{CrossAttn}(q_i,F_{\mathrm{cam}}),
$$

先让 query 从多目视觉中形成主要语义表征。然后只让 LiDAR 更新几何相关变量：

$$
(\Delta\mu_i,\Delta s_i,\Delta a_i)
=
\operatorname{GeoCrossAttn}
(q_i^{\mathrm{vis}},F_{\mathrm{lidar}}).
$$

关键不是“LiDAR attention 权重小一点”，而是通过 head 和梯度路由明确限制职责：

- LiDAR branch 主要更新 Gaussian center、scale、depth confidence、occupied evidence；
- visual branch 更新 open semantic feature、known/new posterior、appearance uncertainty；
- 融合 transformer 更新共享上下文，但最终语义 head 不能只走 LiDAR shortcut。

### 11.4 把 query 真正升级成 entity

至少需要新增：

1. **Class-agnostic entity existence**：这里是否存在一个边界相对完整的 thing。
2. **Query-to-mask / query-to-Gaussian assignment**：一个 entity query 覆盖哪些 voxels/Gaussians。
3. **跨帧一致性**：同一 entity 在相邻帧的 feature、中心、shape 和 ID 保持一致。
4. **Stuff/entity 分流**：道路、墙等用 stuff queries；潜在物体用 entity queries。
5. **允许多个 query 合作表示大物体，同时抑制无意义重复**。

此时它才真正接近 PaSCo / MaskFormer 的 entity 语义，而不是仅仅“Gaussian 的父节点”。

### 11.5 OOD 语义不能只加一个 unknown class

更合理的是给每个 entity/query 同时预测：

- open semantic embedding $z_i^{\mathrm{open}}$，与 VLM text/image space 对齐；
- closed-set known-class posterior $p_i(y\mid\mathrm{known})$；
- known vs. novel posterior $p_i^{\mathrm{novel}}$；
- epistemic / temporal uncertainty $u_i$。

这样 unknown 的判断来自：

```text
它像一个真实 entity
+ 多帧和多传感器都支持其存在
+ 却不能被任何 known prototype 稳定解释
```

而不是把“unknown”当作训练集中另一个固定类别。

### 11.6 OOD-aware query propagation

原始 top-k opacity 应换成综合保留分数：

$$
S_i=
\lambda_o p_i^{\mathrm{occ}}
+\lambda_e p_i^{\mathrm{entity}}
+\lambda_u U_i^{\mathrm{OOD}}
+\lambda_r R_i^{\mathrm{risk}}
+\lambda_t P_i^{\mathrm{persist}}.
$$

其中：

- $p^{\mathrm{occ}}$：是否真的占据空间；
- $p^{\mathrm{entity}}$：是否像独立物体；
- $U^{\mathrm{OOD}}$：是否难以被 known semantics 解释；
- $R^{\mathrm{risk}}$：离 ego path 的距离、碰撞时间等风险；
- $P^{\mathrm{persist}}$：历史上是否稳定存在。

还应当给高 OOD uncertainty 的 query 一个短期“宽限期”：即使当前 opacity 低，也先保留 2–3 帧，让时序证据有机会积累。

距离抑制也不能全局固定，可以：

- stuff query 使用较大 $\delta$；
- entity query 使用较小 $\delta$；
- 高风险或高 OOD query 允许更密集保留；
- 根据预测 physical size 自适应 suppression radius。

---

## 12. 一个适合我们的训练目标草图

虽然当前重点是建模，写出 loss 可以帮助判断每个模块是否真的可训练：

$$
\mathcal L=
\mathcal L_{\mathrm{occ}}
+\lambda_e\mathcal L_{\mathrm{entity}}
+\lambda_d\mathcal L_{\mathrm{render-depth}}
+\lambda_v\mathcal L_{\mathrm{render-VLM}}
+\lambda_k\mathcal L_{\mathrm{known}}
+\lambda_n\mathcal L_{\mathrm{novel}}
+\lambda_t\mathcal L_{\mathrm{temporal}}.
$$

各项含义：

| Loss | 约束对象 | 目的 |
|---|---|---|
| $\mathcal L_{occ}$ | G2V voxel output | free / occupied 与已知语义 |
| $\mathcal L_{entity}$ | entity query / mask | objectness、实例完整性 |
| $\mathcal L_{render-depth}$ | Gaussian geometry | LiDAR depth / ray consistency |
| $\mathcal L_{render-VLM}$ | Gaussian semantic feature | 3D feature 与 2D foundation feature 对齐 |
| $\mathcal L_{known}$ | known semantic head | 正常 closed-set 能力 |
| $\mathcal L_{novel}$ | novelty head | 已知/未知可分性与校准 |
| $\mathcal L_{temporal}$ | propagated query | 位置、feature、identity、novelty 稳定 |

最重要的梯度原则是：

> unknown 区域不应被 known-class CE 强行拉向某个已知 prototype；entity existence 与 known semantic classification 必须解耦。

---

## 13. 如何评价“基于 S2GO 做 OOD occupancy”这条科研路线？

### 13.1 科研上有价值的部分

这条路线的真正新意不应写成：

> “我们在 S2GO 上加了一个 OOD head。”

更有说服力的命题是：

> **Closed-set sparse Gaussian occupancy 的 query retention 与语义共享机制，会系统性地遗忘低证据 novel objects；我们设计 geometry-supported、open-semantic、risk-aware persistent entities，使未知物体的弱证据能够跨帧积累，并最终生成 unknown occupancy。**

对应的核心研究问题是：

1. 如何把 occupancy query 从 spatial anchor 变成真正的 entity hypothesis？
2. 如何解耦 occupied、objectness、known semantics 和 novelty？
3. 如何在固定 query budget 下，保证 novel object 不被时序筛选机制过早删除？
4. Camera 与 LiDAR 如何形成“语义主导—几何校正”的非对称融合？
5. 如何把 VLM 的开放语义投影到 3D Gaussian，再聚合成稳定的 unknown occupancy？

这比单纯换 backbone 或叠加 uncertainty score 更像一个完整方法。

### 13.2 最大风险

最大的风险是把三个尚未解决的问题同时绑在一起：

- S2GO 复现；
- camera–LiDAR fusion；
- open-set entity occupancy。

如果没有可用官方代码，工程风险会很高。因此实验顺序最好是：

```text
1. 先复现 sparse query → Gaussian → G2V 的单帧 closed-set baseline
2. 再验证 temporal query propagation
3. 再加入 camera-dominant LiDAR geometry fusion
4. 最后加入 entity/OOD heads 与 OOD-aware propagation
```

每一步都应有独立的可验证收益，否则最终失败时很难定位问题。

---

## 14. 最值得做的 Ablation

### 14.1 S2GO 表示本身

- independent Gaussians vs. parent query + child Gaussians；
- 1 query 多个 children vs. 更多单 Gaussian queries；
- shared semantics vs. child-specific semantics；
- opacity-aware vs. opacity-free G2V。

### 14.2 Temporal OOD retention

- 不传播；
- opacity top-k；
- opacity + distance；
- entity score；
- entity + OOD uncertainty；
- entity + OOD + risk + grace period。

这里最关键的指标不只是 overall mIoU，而是：

- OOD object recall 随距离的曲线；
- 首次可见到被稳定检测的延迟；
- 短暂遮挡后的 re-detection；
- query survival rate；
- false positive persistence；
- OOD object-level AP / FPR95 / AUROC；
- unknown occupied voxel IoU。

### 14.3 Camera–LiDAR 职责分离

- camera only；
- symmetric fusion；
- camera query → LiDAR cross-attention；
- LiDAR query → camera cross-attention；
- 几何 head 只接 LiDAR、语义 head 只接 camera；
- 加入 modality dropout，检查模型是否偷偷依赖 LiDAR 语义 shortcut。

---

## 15. 读 S2GO 时最容易产生的五个误解

### 误解 1：“一个 query 就是一辆车”

不一定。它是局部高层 anchor，不受 instance GT 约束。

### 误解 2：“Gaussian 直接就是 occupancy”

不是。Gaussian 是中间连续表示，最终还要通过 opacity-aware G2V 得到规则 voxel occupancy。

### 误解 3：“S2GO 是 camera–LiDAR fusion”

不是。原论文正常推理是 camera-only；LiDAR 主要用于 Stage 1 query denoising / geometry pretraining。

### 误解 4：“有 opacity 就能检测 unknown object”

不能。opacity 只反映 primitive 的存在贡献，不等价于 entity objectness，更不等价于 semantic novelty。

### 误解 5：“Temporal query 自然会帮助 OOD”

不一定。原始 top-k opacity 策略可能恰好最先删除弱小、远距离、刚出现的 OOD，必须重新设计 retention。

---

## 16. 最终判断

S2GO 对我们的价值，不在于它已经解决了 OOD，而在于它提供了一个很合适的中间骨架：

```text
稀疏时序状态（query）
        +
连续局部几何（Gaussian bundle）
        +
标准稠密输出（voxel occupancy）
```

它比纯 voxel temporal model 更容易维护 object-like memory，也比完全独立的 Gaussian representation 更容易加入 entity-level reasoning。

但原始 query 还不是 entity，原始 semantic head 还是 closed-set，原始 propagation 也不保护 novel object。因此真正值得做的改造应是：

> **以 S2GO 的 persistent query–Gaussian hierarchy 为 backbone，加入 camera-dominant / LiDAR-geometric asymmetric fusion，把 query 升级为显式 entity hypothesis，并用 open-semantic 与 OOD-aware temporal retention 输出稳定的 unknown occupancy。**

这才是 S2GO 与我们的 open-set occupancy 课题之间最紧密、也最有研究含量的连接。

---

## 17. 术语速查

| 术语 | 本文中的含义 |
|---|---|
| Persistent query | 跨帧传播的稀疏场景状态 |
| Parent query | 生成一组 child Gaussians 的父节点 |
| Child Gaussian | 描述局部 3D 形状的椭球 primitive |
| Opacity | Gaussian 对实际占据/渲染的存在贡献 |
| G2V | Gaussian-to-Voxel，将连续表示转为规则 occupancy |
| Traversal | query 从错误/空空间位置移动到真实表面的能力 |
| Entity | 有独立存在性、范围和时序一致性的物体假设；原始 S2GO 尚未严格具备 |
| Stuff | road、wall、vegetation 等无明确实例边界的背景结构 |
| Open semantic feature | 与 VLM 等开放词汇表征对齐的 feature，而非固定类别 logits |
| Query retention | 在固定预算下决定哪些历史 query 继续传播 |

---

## 18. 主要资料

- [S2GO arXiv](https://arxiv.org/abs/2506.05473)
- [S2GO Project Page](https://jindapark.github.io/projects/s2go/)
- [S2GO OpenReview / ICLR 2026](https://openreview.net/forum?id=z8ggdMlSco)
- [GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_GaussianFormer-2_Probabilistic_Gaussian_Superposition_for_Efficient_3D_Occupancy_Prediction_CVPR_2025_paper.html)
- [GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](https://openaccess.thecvf.com/content/CVPR2025/html/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.html)

> 注：表格中的性能与消融数值来自 S2GO 论文；不同数据集、体素范围、输入分辨率和硬件下的数值不可直接横向比较。截至本文整理时，官方项目页提供论文资料，但没有看到可直接使用的官方代码入口，因此复现计划需要预留额外工程成本。
