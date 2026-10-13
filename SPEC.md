# SPEC — Gromov-Wasserstein Factorization (GWF) 复现技术契约

> **状态**：**v1.0 已冻结**（评审通过 2026-10-12；此后允许创建 `gwf/*.py`、`tests/*.py`、`scripts/*.py`）。
> **日期**：2026-10-12
> **依据论文**：*Representing Graphs via Gromov-Wasserstein Factorization*, Xu, Liu, Luo, Carin, IEEE TPAMI 2022, DOI 10.1109/TPAMI.2022.3153126
> **权威源**（凡冲突以此为准）：`paper/GW works/.pdf/[main] 通过Gromov-Wasserstein因子分解表示图.tex`
> **辅助源**：`paper/GW works/[main] Representing_Graphs_via_Gromov-Wasserstein_Factorization.md`、`paper/GW works/[Intro] Representing_Graphs_via_Gromov-Wasserstein_Factorization.md`、`paper/GW works/.pdf/abtract.md`
> **任务提示词**：`code/prompt.md`（本契约是它的 §6 落地）
> **依据 git commit**：`9fb9adb`

---

## 0. 前置说明与变更记录

### 0.1 阅读规则

1. 本文件是后续所有实现的**唯一技术契约**：代码里每个函数、每个超参都必须能在本文件的对应条目找到出处（论文式号/算法号/prompt §号）。凡本文件未覆盖的实现细节，**先补本文件并记 Changelog，再写代码**。
2. 冲突优先级：**LaTeX 源（论文公式）> prompt.md > 本文件旧版本**。冲突一律记入 `NOTES.md` 与本文件 Changelog。
3. 引用标记约定：`[式 N]`=论文式号，`[Alg.N]`=论文算法号，`[Thm.4.1]`=包络定理，`[§11-N]`=prompt.md 第 11 节第 N 条歧义，`[决策]`=本文件自主决定（须给依据），`[待确认]`=尚未锁定、需实验判定。

### 0.2 变更记录（Changelog）

| 版本 | 日期 | 变更内容 | 依据 |
| :-- | :-- | :-- | :-- |
| v1.0 | 2026-10-12 | 初稿，十二章节齐全，待评审 | prompt.md §6 |
| v1.0（冻结） | 2026-10-12 | 评审通过，冻结 v1.0 | 用户批准 |

---

## 1. 目标与范围

### 1.1 目标（三条主线，缺一不可）[prompt §0]

在 `code/gwf/` 下用 **PyTorch** 从零复现 GWF，并用真实图数据集验证图聚类（Rand-Index）与图分类（Accuracy）指标。

1. **GWF（直推式 / transductive）**：可学习图因子 $U_{1:K}$ + 每图权重 $z_i$（softmax 后为 $\lambda_i$）[式 9, 12, 13]。
2. **GWD 求解器两个版本**：`GWD_PPA` [Alg.1] 与 `GWD_BADMM` [Alg.2]，可插拔互换。
3. **GWF-GNN（归纳式 / inductive）**：GNN 编码器 $h_\theta$ 替代 $z_i$，构成"GNN 编码 + GWF 解码"自编码框架 [式 15]。

### 1.2 扩展项（分阶段，允许后置）[prompt §0]

| 优先级 | 扩展 | 论文出处 |
| :-- | :-- | :-- |
| P1 | FGW / 节点属性版本 | [式 25, 26] |
| P2 | 半监督分类正则 | [式 24] |
| P3 | 交替优化 vs 联合优化对比（主线内必须做） | [Alg.4] 后注 |

### 1.3 Out of Scope（明确不做）

- 不实现论文引用的基线方法（InfoGraph、GCL、FGWK、GWB-KM、GDL、各种图核）的完整复现；仅实现熵正则化 GW 基线用于 Phase 3 定性对照 [§9 Phase 3 第 7 条]。
- 不做 sliced GW / recursive GW 等未来工作加速。
- 不追求与官方仓库逐行一致（官方仓库仅作交叉核对，不作抄写来源）。

---

## 2. 记号与张量契约

### 2.1 符号表（全局唯一口径）

| 符号 | 含义 | shape | dtype | 来源 |
| :-- | :-- | :-- | :-- | :-- |
| $C_i$ | 第 $i$ 个观测图邻接/距离矩阵 | $(N_i, N_i)$ | float32 | [式 6] |
| $\mu_i$ | 观测图节点分布（均匀） | $(N_i,)$ | float32，$\mu_i=\frac{1}{N_i}\mathbf{1}$ | [式 5 后文] |
| $V_k$ | 第 $k$ 因子 logit（可学习） | $(N_k, N_k)$ | float32 | [式 12] |
| $U_k=f(V_k)$ | 第 $k$ 因子（sigmoid 后） | $(N_k, N_k)$ | float32，$0\le U_k\le1$ | [式 12] |
| $\mu_k$ | 因子节点分布（均匀，固定） | $(N_k,)$ | float32，$\mu_k=\frac{1}{N_k}\mathbf{1}$ | [§3 约定 3] |
| $z_i$ | 图的自由权重参数（可学习） | $(K,)$ | float32 | [式 13] |
| $\lambda_i=g(z_i)$ | softmax 后权重 | $(K,)$，$\sum_k\lambda_{ik}=1$ | float32 | [式 13] |
| $T_k$ | $B$（X 侧）到 $U_k$（Y 侧）的最优传输矩阵 | $(N_b, N_k)$ | float32 | [Alg.3] |
| $B^{(l)}$ | 第 $l$ 步 GW 重心（重建图） | $(N_b, N_b)$，$N_b=N_i$，$B^{(0)}=C_i$ | float32 | [Alg.3] |
| $H_k$ | 因子节点可学习嵌入（仅节点属性版） | $(N_k, D)$ | float32 | [式 26] |
| $C_{XY}$ | GW 二次型线性项 | $(N_X, N_Y)$ | float32 | [式 7] |
| $F_i$ | 观测图节点属性 | $(N_i, D)$ 或 `None` | float32 | [式 25] |

### 2.2 硬约定（写进代码 docstring，违反即测试失败）

1. **重心维度与观测图一致**：`B.shape == C_i.shape == (N_i, N_i)`；因此 $T_k$ 是 $(N_i, N_k)$，**$T_k$ 的 size 依赖 $i$，不能跨图共享，不能预分配全局 $T$ 张量**。[§3 约定 1][§12 坑 5]
2. **传输多面体**：$T_k\ge0$，$T_k\mathbf{1}_{N_k}=\mu_i$（行边际，X 侧），$T_k^\top\mathbf{1}_{N_i}=\mu_k$（列边际，Y 侧）。X 侧 = $B$ / 观测图，Y 侧 = $U_k$。[§3 约定 2]
3. **device/dtype**：训练默认 float32；`gradcheck` 与数值断言处用 float64。默认设备 `mps`，遇不支持的算子退 `cpu`，**两个设备结果必须一致**（差异 `<1e-6`），实际设备记入 `NOTES.md`。[§2 环境]
4. **索引方向**：$T_k U_k^\top T_k^\top$ vs $T_k^\top U_k T_k$ 两种写法只在 $T$ 的行列约定不同时等价。本文件默认采用**论文式 21 形式** $B=\frac{1}{\mu_b\mu_b^\top}\sum_k\lambda_k T_k U_k^\top T_k^\top$，配合 $T_k\in\Pi(\mu_b,\mu_k)$（$T_k$: $B\to U_k$）。**最终以 Phase 4 同尺寸恒等测试锁定**，见 §9 条目 1、§11 Phase 4。[§3 约定 4]
5. **均匀分布假设**：$\mu$ 全为 $1/N$；一旦引入带权节点分布，列边际 $\mu_k$ 必须同步改，否则边际测试静默失败。[§12 坑 7]

---

## 3. 数据规格

### 3.1 数据集清单与统计期望值（来自论文表 7.1）[§5.3]

| 数据集 | #graphs | #classes | 节点属性维度 D | 平均节点数 | 平均边数 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| AIDS | 2000 | 2 | 4 | 15.69 | 16.20 |
| PROTEIN | 1113 | 2 | 1 | 39.06 | 72.82 |
| PROTEIN-F | 1113 | 2 | 29 | 39.06 | 72.82 |
| ENZYMES | 600 | 6 | 18 | 32.63 | 62.14 |
| MUTAG | 188 | 2 | 7 | 17.93 | 19.79 |
| PTC-MR | 344 | 2 | 18 | 14.29 | 14.69 |
| IMDB-B | 1000 | 2 | — | 19.77 | 96.53 |
| IMDB-M | 1500 | 3 | — | 13.00 | 65.94 |

> 统计校验容差：**±0.5%**（相对误差），不满足先修加载，不训练。[§9 Phase 2]

### 3.2 加载规则

- 首选 `torch_geometric.datasets.TUDataset(root='data/', name=<name>)`；若 PyG 不可用，按 TUDataset 原始格式自行解析（`*_A.txt`、`*_graph_indicator.txt`、`*_graph_labels.txt`、`*_node_labels.txt`、`*_node_attributes.txt`）。[§8.1]
- 下载地址：<https://chrsmrrs.github.io/datasets/>。
- 统一接口（仅签名，非实现）[§8.1]：

```
load_dataset(name) -> list[Graph], torch.LongTensor
# Graph 数据结构：C (N_i,N_i) float32 邻接矩阵；F (N_i,D) float32 或 None；y int 类别标签
```

### 3.3 预处理规则

1. **邻接对称化**：可选开关，默认 `False`（保持原始有向/无向结构；论文对无向图数据集已天然对称）。[§8.2][决策]
2. **归一化到 $[0,1]$**：$C_i \leftarrow C_i / \max(C_i)$（若 $\max>0$；全零图保持全零）。邻接矩阵元素已在 $[0,1]$，此步保证含权图（如经权值调制）落入 $[0,1]$。默认开启。[§8.2][式 3.1 $C\in[0,1]^{N\times N}$]
3. **对角置 0**：`C.diagonal().fill_(0)`，默认开启。[§8.2]
4. **离散节点标签 → one-hot**：MUTAG、PTC-MR 等；离散标签取值集合由数据决定（MUTAG 7 类标签 → 7 维 one-hot）。[§8.2][式 25 后文]
5. **无属性图（IMDB-B/M）特征构造**：两种变体（全 1 向量 / 节点度），默认值见 §9 条目 5。[§8.2]
6. **图因子尺寸 $N_k$**：见 §9 条目 2（默认 10，可配 `--factor-nodes`）。

### 3.4 数据校验

`check_dataset_stats(name, graphs)` 将解析结果与 §3.1 表格逐列比对（#graphs / #classes / 平均节点数 / 平均边数），输出比对表；任一项超出 ±0.5% 即抛异常。[§9 Phase 2]

---

## 4. 核心算子规格

### 4.1 GW 差异二次型（$p=2$）[式 6, 7]

给定 $C_X\in\mathbb{R}^{N_X\times N_X}$、$\mu_X$、$C_Y$、$\mu_Y$：

$$
d_{gw}^2(C_X,C_Y)=\min_{T\in\Pi(\mu_X,\mu_Y)}\langle C_{XY}-2C_X T C_Y^\top,\ T\rangle
$$

$$
C_{XY}=(C_X\odot C_X)\mu_X\mathbf{1}_{N_Y}^\top+\mathbf{1}_{N_X}\mu_Y^\top(C_Y\odot C_Y)^\top
$$

- **$p=2$ 硬断言**：代码中 `assert p == 2`；$(C\odot C)$ 仅在 $p=2$ 成立，改 $p$ 需重推。[§12 坑 2]
- **shape 校验**：$C_{XY}$ 为 $(N_X,N_Y)$；两项外积方向见下伪代码，错误会被 §11 Phase 3 第 3 条（$C_X=C_Y\Rightarrow d_{gw}=0,\ T=\mathrm{diag}(\mu)$）兜底。[§12 坑 1]

$C_{XY}$ 构造伪代码（无实现）：

```
C_XY = (C_X ⊙ C_X) @ mu_X[:,None] @ ones(1, N_Y)   # 项1：(N_X,1)@(1,N_Y)
     + ones(N_X,1) @ mu_Y[None,:] @ (C_Y ⊙ C_Y).T   # 项2：(N_X,1)@(1,N_Y)@(N_Y,N_Y)
```

### 4.2 GWD_PPA [Alg.1]

- **入参**：`C_X (N_X,N_X), C_Y (N_Y,N_Y), mu_X (N_X,), mu_Y (N_Y,), M int, gamma float, cost_extra (N_X,N_Y)|None`。
- **出参**：`T (N_X,N_Y)`，满足边际约束（容差见 §11 Phase 3）；可选返回 $d_{gw}$。
- **超参**：$\gamma_{\text{PPA}}=0.01$，$M=20$。[§7 超参][论文 §7.2]

伪代码：

```
C_XY = 构造如 §4.1（若有 cost_extra：C_XY = C_XY + cost_extra）   # [式 25] FGW 复用
T = mu_X[:,None] @ mu_Y[None,:]                                   # T^(0)
a = mu_X
for m in 0..M-1:
    Phi = exp( -(C_XY - 2*C_X @ T @ C_Y.T) / gamma ) * T         # 逐元素；exp 前平移
    b = mu_Y / (Phi.T @ a + eps)                                  # Sinkhorn-Knopp
    a = mu_X / (Phi @ b + eps)
    T = diag(a) @ Phi @ diag(b)
return T
```

**数值稳定策略**（强制）[§4.2 实现要点][§11 条目 10][§12 坑 6]：
- `Phi = exp(-(...)/gamma - shift) * T`，`shift = max(整个括号矩阵)`（对 $\gamma=0.01$ 时 $1/\gamma=100$ 必做）。
- `b`、`a` 分母加 `eps=1e-12`。
- 中间量 `Phi, b, a, T` 每一步断言 `torch.isfinite(...).all()`。
- Sinkhorn 比例因子可吸收常数平移，$T$ 不变（等价性证明式说明写入 `NOTES.md`）[§11 条目 8]。

### 4.3 GWD_BADMM [Alg.2]

- **入参/出参**：同 §4.2。
- **超参**：$\gamma_{\text{BADMM}}=1$，$M=20$。[§7 超参]

伪代码：

```
C_XY = 构造如 §4.1（若有 cost_extra：C_XY += cost_extra）
T = mu_X[:,None] @ mu_Y[None,:];  Z = zeros(N_X, N_Y)
for m in 0..M-1:
    Phi1 = exp( +(2*C_X.T @ T @ C_Y + Z)/gamma ) * T            # exp(+) 必做最大值平移
    S    = Phi1 * ( mu_Y / (Phi1.T @ ones(N_X)) )[None,:]        # 列归一化
    Phi2 = exp( -(C_XY - 2*C_X @ S @ C_Y.T + Z)/gamma ) * S      # 平移
    T    = ( mu_X / (Phi2 @ ones(N_Y)) )[:,None] * Phi2          # 行归一化
    Z    = Z + gamma * (T - S)
return T
```

**数值稳定策略**：同 §4.2；特别地 `Phi1` 的 `exp(+)` **必须**最大值平移（$\gamma=1$ 时仍可能上溢）。[§4.3 实现要点]

### 4.4 GW 重心 GWB [Alg.3][式 21]

- **入参**：`U_list [(N_k,N_k)]*K`（或 $(K,N_k,N_k)$ 堆叠，因子同尺寸时）、`lam (K,)`、`B0 (N_b,N_b)`、`mu_b (N_b,)`、`L int`、`gwd in {'ppa','badmm'}`、`**kw`（含 `M, gamma, cost_extra`）。
- **出参**：`(B (N_b,N_b), T_list [(N_b,N_k)]*K)`。
- **超参**：$L=5$。[§7 超参]

伪代码：

```
for l in 0..L-1:
    for k in 1..K:
        T_k = GWD(B^(l), U_k)                    # X 侧 = B^(l)，Y 侧 = U_k
    B^(l+1) = (1 / (mu_b[:,None] @ mu_b[None,:])) * sum_k lam_k * (T_k @ U_k.T @ T_k.T)
return B^(L), T_{1:K}^(L)
```

**实现约束**：
- 闭式更新来自"给定 $T_k$ 时 $\min_B$"的一步解析解；**不得在对角/归一化上取巧**。$\mu_b\mu_b^\top$ 是 $(N_b,N_b)$ 外积矩阵，**逐元素相除**（`B * (1/outer)`），不是矩阵逆。[§4.4][§12 坑 4]
- $B^{(l)}$ 投影/裁剪到 $[0,1]$：作为**开关**，默认关闭；是否影响结果记入 `NOTES.md`。[§4.4]
- 可选优化：$K$ 个 GWD 调用 batch 化（因子同尺寸堆成 $(K,N_k,N_k)$）；**先在循环版把数值校验做对，再做向量化**，两者输出误差 `<1e-8`。[§4.4]

---

## 5. 模型规格

### 5.1 可学习参数清单

| 参数 | shape | 初始化 | 参与梯度 | 来源 |
| :-- | :-- | :-- | :-- | :-- |
| $V_{1:K}$（logit） | $(K, N_k, N_k)$ | 预热 `logit(0.98*C_k+0.01)` | 是 | [式 12][Alg.4] |
| $z_{1:I}$（logit） | $(I, K)$ | `N(0, 0.01)` | 是 | [式 13][§11 条目 6] |
| $H_{1:K}$（节点属性版） | $(K, N_k, D)$ | 随机 或 取自数据 | 是 | [式 26] |
| $\theta$（GNN 编码器） | 架构依赖 | 默认 | 是 | [式 15] |
| $T_{1:K}^{(L)}$ | 前向中间量 | — | **否（必须 detach）** | [Thm.4.1] |

### 5.2 重参数化映射 [式 12, 13]

$$
U_k = f(V_k) = \mathrm{sigmoid}(V_k),\qquad \lambda_i = g(z_i) = \mathrm{softmax}(z_i)
$$

- 可学习的是 **logit 空间的 $V,z$**（以及 $H,\theta$），**不是** $U,\lambda$；优化器只更新 logit。[§4.6]
- 变量定义域自动满足 $0\le U\le1$、$\lambda\in\Delta^{K-1}$，**不手动加投影**（除 §4.4 重心裁剪开关）。[§4.6]

### 5.3 前向计算图（单图 $i$）

```
V -> U = sigmoid(V)                              # 所有因子
z_i -> lam_i = softmax(z_i)
B^0 = C_i;  mu_b = mu_i
B^L, T_{1:K}^L = GWB(U, lam_i, B^0, mu_b, L, gwd)
loss_i = || B^L - C_i ||_F^2                      # [式 27]，见 §6
```

**GWF-GNN 变体** [式 15]：将 `z_i -> lam_i` 替换为 `lam_i = softmax(h_theta(C_i))`，$h_\theta$ 输出维度 $=K$。无属性图输入特征见 §9 条目 5。

### 5.4 包络定理 / detach 规则 [Thm.4.1][§4.7]

- 前向结束时拿到 $T_{1:K}^{(L)}$，随后在**目标函数的重建项**中把它当常数：

$$
\mathcal{L}_i=\Big\|\frac{1}{\mu_b\mu_b^\top}\sum_k g_k(z_i)\,T_k^{(L)}\,f(V_k)\,(T_k^{(L)})^\top - C_i\Big\|_F^2
$$

- 实现语义：**前向仍用未 detach 的 $T$ 计算 $B$**（保证 $B$ 数值正确）；**只有进入重建损失时把 $T_k$ 视为常数**，用 `T_k.detach()`。两者数值相同，但必须 detach，否则会沿 Sinkhorn 迭代反传，训练时间爆炸。[§4.7]
- **强制断言**：`assert not T_k.requires_grad`（在进入损失前）[§12 坑 3]。
- 文档标注：包络定理适用前提是 $T_k$ 为对应最优解（$M$ 步后成立近似）；"是否 detach 的耗时对比"实验写入 `NOTES.md`。[§4.7]

---

## 6. 目标函数与梯度

### 6.1 损失定义 [式 27]

$$
\mathcal{L}=\sum_{i=1}^{I}\|B_i^{(L)}-C_i\|_F^2
$$

- **可选归一化**：除以 $N_i^2$（开关 `--loss-normalize`，默认 `False`=原始 $\|\cdot\|_F^2$）。[§11 条目 7]
- $d_{\text{loss}}$ 只用 Frobenius 范数（论文明确：比 GW 差异做 loss 更快且效果相当）。[式 27 后文]
- 节点属性版（FGW）：$\mathcal{L}=\|B^{(L)}-C_i\|_F^2+\|F_b^{(L)}-F_i\|_F^2$。[式 25 后文]

### 6.2 梯度计算路径

- $\partial\mathcal{L}/\partial V_k$：经 $U_k=f(V_k)$ 与 $T_k U_k^\top T_k^\top$（$T_k$ detach）。
- $\partial\mathcal{L}/\partial z_i$：经 $\lambda_i=g(z_i)$ 与 $\lambda_k$ 系数。
- 不反传进 GWD 的 Sinkhorn 迭代（见 §5.4）。

### 6.3 梯度检查方案 [§9 Phase 5 第 2 条]

- 用 `torch.autograd.gradcheck`：double 精度，$N_k=4$，$M=3$，$L=2$，$K=2$，$I=2$。
- 校验 $\partial\mathcal{L}/\partial V$ 与数值梯度一致，容差 `1e-4`。
- 同时断言：detach 后梯度与未 detach 梯度**数值近似一致但更快**；耗时对比写入 `NOTES.md`。

---

## 7. 训练规格

### 7.1 超参（论文值，全部走 config，不写死）[§4.8][论文 §7.2]

| 超参 | 值 | 取值范围 | 来源 |
| :-- | :-- | :-- | :-- |
| $K$ | 15 / 30 / 150 | `{15,30,150}` | 论文 §7.2 |
| $M$（GWD 迭代） | 20 | 正整数 | 论文 §7.2 |
| $L$（GWB 迭代） | 5 | 正整数 | 论文 §7.2 |
| $\gamma_{\text{PPA}}$ | 0.01 | 正数 | 论文 §7.2 |
| $\gamma_{\text{BADMM}}$ | 1 | 正数 | 论文 §7.2 |
| Adam lr | 0.01 | 正数 | 论文 §7.2 |
| epochs | 10 | 正整数 | 论文 §7.2 |
| batch | 单图（per-graph） | `per-graph` / `mini-batch` | §11 条目 3 |
| optim | `joint` / `alternating` | 两者都实现 | §4.8 |
| $N_k$（因子尺寸） | 10 | `{5,10,15,20}` | §11 条目 2 |
| GNN 层数 | 5 | `{5,10,15}` | 论文 §7.2 |

### 7.2 初始化 [Alg.4][论文 §7.2]

1. **预热**：从数据集随机取 $K$ 个图 $C_{1..K}$，$V_k=\mathrm{logit}(0.98\,C_k+0.01)$（对邻接矩阵逐元素做 sigmoid 反函数；若元素恰为 0/1，`logit` 需裁剪到 `[eps, 1-eps]` 内再求反，`eps=1e-6`）[决策]。必须实现。[§4.8]
2. **$z_{1:I}$ 随机初始化**：`N(0, 0.01)`；初始化后检查 $\lambda_i=\mathrm{softmax}(z_i)$ 应接近均匀（各分量 ≈ $1/K$）。[§11 条目 6]
3. **因子节点分布**：均匀 $\mu_k=1/N_k$，固定不学习。[§4.8]

### 7.3 每轮更新顺序 [Alg.4]

```
for epoch in 1..E:
    for i in 1..I:
        B^0 = C_i;  mu_b = mu_i
        B^L, T_{1:K}^L = GWB(f(V), g(z_i), B^0, mu_b)
        L_i = ||B^L - C_i||_F^2
        固定 T_{1:K}^L，算 ∇_V L_i、∇_{z_i} L_i
        Adam 更新 V 与 z_i
```

### 7.4 joint vs alternating 精确定义 [§4.8][§11 条目 4]

- **joint**：每个图 $i$ 上同时更新 $V$ 与 $z_i$（一个 Adam step 同时作用两者）。
- **alternating（默认）**：先冻结 $V$ 更新 $z_i$（1 step），再冻结 $z_i$ 更新 $V$（1 step）。顺序固定为"先 z 后 V"。
- **inner step 消融**：alternating 每轮做 1 / 3 / 5 步的消融实验，默认 1 步。[§11 条目 4]

---

## 8. 评估规格

### 8.1 图聚类（Rand-Index）[§7.3 协议]

1. 训练 GWF / GWF-GNN → 取表示：**统一用 $\lambda_i=\mathrm{softmax}(z_i)$**（GWF-GNN 用 softmax 后 GNN 输出）。[§11 条目 9]
2. K-means：簇数 = 类别数；`n_init>=10`，固定 `random_state`（与全局 seed 绑定）。[§12 坑 8]
3. 计算 Rand-Index（与真实标签比较），报告百分比。

### 8.2 图分类（Accuracy）[§7.4 协议]

1. 用学到的表示（$K=30$ 或 150）→ 10 折交叉验证。
2. SVM 分类（LIBSVM 风格，`sklearn.svm.SVC`）；$C$ 从 $\{10^{-3},10^{-2},\dots,10^{2},10^{3}\}$ 中**仅在训练折内交叉验证选取**，严禁在测试折上泄露。[§12 坑 9]
3. 报告 accuracy（均值 ± 折间 std）。

### 8.3 随机性口径 [§7 随机性][§12 坑 11]

- 每个配置 **≥5 个随机种子**；固定种子记录进结果 JSON。
- **std 口径与论文一致**：分类表 = 10 折 CV 的折间 std；聚类表 = 多次运行（不同 seed）的 std。报告注明口径。
- 所有超参、设备、耗时落盘。

### 8.4 结果 JSON 字段 [§7 运行日志]

```
results/<dataset>/<model>_K<K>_<optim>_seed<S>.json
字段（至少）：dataset, model, K, M, L, gamma, lr, epochs, optim, seed,
           metric, value, std, runtime_sec, device, git_commit
```

---

## 9. 歧义登记表（§11 十条逐条落实）

| # | 歧义项 | 默认取值 | 判定实验 | 判定标准 | 状态 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | $C_{XY}$ 转置约定 与 $T_kU_k^\top T_k^\top$ vs $T_k^\top U_k T_k$ | 论文式 21 形式 $T_k U_k^\top T_k^\top$，$T_k\in\Pi(\mu_b,\mu_k)$（$T_k$: $B\to U_k$） | Phase 4 测试 5：同尺寸 $N_k=N_b$ 恒等测试（$K=1,\lambda=[1],U_1=C$） | 能通过 $d_{gw}(\hat B,U_1)<\epsilon$ 的形式为正确；若两者都过，改 $N_k\ne N_b$ + 前向损失更低的，并用 $d_{gw}(B,C_i)$ 判据 | 未定→Phase 4 |
| 2 | 因子尺寸 $N_k$ | $N_k=10$（小图可 `min_i N_i`） | $N_k\in\{5,10,15,20\}$ 敏感性实验 | 记录 RI/损失随 $N_k$ 曲线，锁定默认值 | 未定→Phase 8 |
| 3 | per-graph vs mini-batch | per-graph（最忠实 Alg.4） | 实现小 batch（$B=20$）并对比 | 记录损失/RI/耗时差异，锁定默认 | 未定→Phase 5 后 |
| 4 | alternating inner step 数与顺序 | update z(1 step) → update V(1 step) | 1/3/5 step 消融 | 记录 RI 曲线，锁定默认 | 未定→Phase 5 |
| 5 | IMDB 无属性图 GNN 输入 | 全 1 向量 | 全 1 vs 节点度 消融 | 记录分类/聚类，锁定默认 | 未定→Phase 7 |
| 6 | $z$ 初始化分布与尺度 | `N(0, 0.01)` | 检查初始化后 $\lambda_i\approx$ 均匀 | 各分量 $\approx 1/K$（容差 `1e-2`） | 已定（默认值） |
| 7 | 损失是否按 $N_i^2$ 归一化 | 原始 $\|\cdot\|_F^2$（不归一化） | 归一化开关对比 | 大图稳定性 + 指标对比，锁定默认 | 未定→Phase 5 |
| 8 | Sinkhorn 温度/平移等价性 | 平移不改变 $T$ | 数学证明式说明 | 证明式写入 `NOTES.md`；测试断言平移前后 $T$ 相等（容差 `1e-8`） | 已定（证明） |
| 9 | K-means 表示 | $\lambda_i$（softmax 后） | $\lambda_i$ vs $z_i$ 对照 | 记录 RI，锁定默认 | 已定（默认） |
| 10 | PPA $\gamma=0.01$ 数值风险 | log-space / 最大值平移 | 断言中间量 `isfinite`、无 NaN | §11 Phase 3 第 6 条全绿 | 已定（稳定策略） |

> 状态"已定（默认值/证明）"表示按默认执行并在 `NOTES.md` 记录；"未定"表示按默认跑通主线后，用对应判定实验锁定并**回改本表 + Changelog**。

---

## 10. 溯源表（论文 ↔ SPEC ↔ 代码 ↔ 测试）

| 论文出处 | 内容 | SPEC 条目 | 计划函数/模块 | 验证测试 id | 状态 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 式 5 | GW 距离定义 | §4.1 | —（背景） | — | — |
| 式 6 | GW 差异（$p$ 次） | §4.1 | `transport.gw_discrepancy` | `test_gwd::test_symmetry` | 待实现 |
| 式 7 | $p=2$ 二次型 + $C_{XY}$ | §4.1 | `transport._make_cxy` | `test_gwd::test_symmetry` | 待实现 |
| 式 8 | GW 重心定义 | §4.4 | `barycenter.gwb` | `test_gwb::test_single_factor` | 待实现 |
| 式 9 | GWF 目标 | §5, §6 | `model.GWF` | `test_grad::test_gradcheck` | 待实现 |
| 式 12 | $U=\mathrm{sigmoid}(V)$ | §5.2 | `model.GWF` | `test_grad::test_reparam` | 待实现 |
| 式 13 | $\lambda=\mathrm{softmax}(z)$ | §5.2 | `model.GWF` | `test_grad::test_reparam` | 待实现 |
| 式 15 | GWF-GNN | §5.3 | `model.GWFGNN` | Phase 7 验证 | 待实现 |
| 式 16 | 重构目标（$L,M$ 近似） | §6 | `model.GWF` | `test_grad::test_gradcheck` | 待实现 |
| 式 21 | 重心闭式更新 | §4.4 | `barycenter.gwb` | `test_gwb::test_index_direction` | 待实现 |
| 式 24 | 半监督正则 | §1.2（P2） | `model` 扩展 | Phase 8 | 待实现 |
| 式 25 | FGW | §4.2（cost_extra）、§1.2（P1） | `transport` + `model.FGW` | Phase 8 | 待实现 |
| 式 26 | $F_b$ 特征更新 | §1.2（P1）、§6.1 | `model.FGW` | Phase 8 | 待实现 |
| 式 27 | Frobenius 损失 | §6.1 | `model.GWF.loss` | `test_grad::test_gradcheck` | 待实现 |
| Alg.1 | GWD_PPA | §4.2 | `transport.gwd_ppa` | `test_gwd::*` | 待实现 |
| Alg.2 | GWD_BADMM | §4.3 | `transport.gwd_badmm` | `test_gwd::*` | 待实现 |
| Alg.3 | GWB$^{(L)}$ | §4.4 | `barycenter.gwb` | `test_gwb::*` | 待实现 |
| Alg.4 | GWF 训练 | §7 | `train.train_gwf` | Phase 5 冒烟 | 待实现 |
| Thm.4.1 | 包络定理 detach | §5.4 | `train`（detach） | `test_grad::test_detach` | 待实现 |

> 状态列在 Phase 9 交付时须全部填满（"已实现 + 测试 id 通过"）。

---

## 11. 验收标准（机器可判定）

### 11.1 数值容差汇总

| 项 | 容差 |
| :-- | :-- |
| $T$ 非负 | $T\ge -1\mathrm{e}{-12}$ |
| $T$ 行/列边际 | $\|T\mathbf{1}-\mu_X\|_\infty<1\mathrm{e}{-6}$、$\|T^\top\mathbf{1}-\mu_Y\|_\infty<1\mathrm{e}{-6}$ |
| 置换等变性（$d_{gw}$ 不变） | `1e-6` |
| $d_{gw}$ 随 $M$ 单调不增 | 允许 `1e-8` 抖动 |
| 循环 vs 向量化 GWB 输出误差 | `<1e-8` |
| gradcheck | `1e-4` |
| 重参数化（$U\in[0,1]$、$\lambda$ 行和=1） | `1e-6` |
| 数据统计校验 | ±0.5% |

### 11.2 各阶段通过门槛

| Phase | 通过标准 |
| :-- | :-- |
| 1 | §6.2 十二节齐全、§6.4 自检全过、§11 十条有结论/判定设计、溯源表覆盖 §4 全部公式/算法、评审确认 |
| 2 | MUTAG/PROTEIN/ENZYMES 统计量 ±0.5% 一致；`tests/test_data.py` 全绿 |
| 3 | GWD 测试 1–6 全绿；测试 7（PPA vs BADMM vs 熵正则基线）定性趋势与论文一致 |
| 4 | GWB 测试 1–6 全绿；索引方向锁定并记 `NOTES.md` |
| 5 | 冒烟测试（RI>0.9、重建误差下降≥50%）、gradcheck 通过、过拟合、重参数化、交替 vs 联合；MUTAG $K=15$ PPA 10 epoch RI 与 69.2 偏差 ≤3 |
| 6 | 聚类表 7 数据集 × 2 模型（$K\in\{15,30\}$，5 seeds）多数落在 ±3 个百分点内；偏差大者给出原因假设 |
| 7 | GWF-GNN 至少一个数据集优于 GWF；分类表偏差 ≤3 个百分点 |
| 8 | FGW 不劣于无属性版；$N_k/K/M/L$ 敏感性趋势；t-SNE + 因子热力图 |
| 9 | 全部交付物齐全，`pytest -q` 全绿 |

### 11.3 复现对照表（§5 验收对照，写入 REPORT）

聚类 RI（%）：AIDS 99.0±0.4 / PROTEIN 59.8±0.2 / ENZYMES 72.3±0.3 / MUTAG 69.2±2.0 / PTC-MR 52.5±1.1 / IMDB-B 51.8±0.7 / IMDB-M 55.9±0.4（GWF$_{\text{PPA}}$）；BADMM 与 GWF-GNN 列见 prompt.md §5.1。

分类 Accuracy（%）：PROTEINS-F 73.89 / ENZYMES 73.06 / MUTAG 82.25 / PTC-MR 62.27 / IMDB-B 60.93 / IMDB-M 39.97（GWF$_{\text{PPA}}$ K=30）；其余行见 prompt.md §5.2。

**验收容差**：分类/聚类与论文值偏差在 **±3 个百分点**内视为"复现成功"；报告同时给出"是否落在论文 ± 标准差范围内"。

---

## 12. 接口清单（签名级，非实现）

```
# gwf/data.py
load_dataset(name) -> list[Graph], torch.LongTensor
check_dataset_stats(name, graphs) -> dict              # 与 §3.1 逐列比对

# gwf/transport.py
gwd_ppa(C_X, C_Y, mu_X, mu_Y, M, gamma, cost_extra=None) -> T
gwd_badmm(C_X, C_Y, mu_X, mu_Y, M, gamma, cost_extra=None) -> T
gw_discrepancy(C_X, C_Y, mu_X, mu_Y, M, gamma, gwd='ppa', cost_extra=None) -> float
entropic_gw_baseline(C_X, C_Y, mu_X, mu_Y, ...) -> T   # 熵正则基线（Phase 3 对照）

# gwf/barycenter.py
gwb(U_list, lam, B0, mu_b, L, gwd='ppa', **kw) -> (B, T_list)

# gwf/model.py
GWF(K, N_k, ...)                                       # V (K,N_k,N_k), z (I,K)
GWFGNN(K, encoder, ...)                                # V + GNN theta
FGW(K, N_k, D, ...)                                    # + H (K,N_k,D)

# gwf/train.py
train_gwf(model, graphs, cfg, optim='alternating') -> model   # 含 envelope detach

# gwf/evaluate.py
kmeans_ri(embeddings, labels, n_clusters, seed) -> float
svm_10fold(embeddings, labels, C_grid, seed) -> (mean, std)

# gwf/utils.py
set_seed(seed);  Timer;  save_json(...);  load_json(...)
```

**依赖关系**：`data → transport → barycenter → model → train → evaluate`；`utils` 被所有模块依赖。

**创建顺序约束**：`SPEC.md` 必须先于 `gwf/`、`tests/`、`scripts/` 三个目录存在；`gwf/` 及以下条目只能在 SPEC 评审通过后创建。[§10 创建顺序]

---

## 附：§6.4 SPEC 定稿自检清单

- [x] 十二条章节齐全，无空章节。
- [x] 所有超参都有具体默认数值，无"待定"。
- [x] 每个函数签名都标了输入/输出 shape 与 dtype（§2.1、§4、§12）。
- [x] 每个公式条目都标了论文出处或 `[决策]`。
- [x] §11 十条全部有"默认取值 + 判定实验 + 判定标准"（§9）。
- [x] 索引方向判定测试已写明断言与容差（§9 条目 1、§11.1）。
- [x] detach 规则、数值稳定策略、eps 位置已写明（§4.2/4.3/5.4）。
- [x] 数据统计校验期望值已写入（§3.1）。
- [x] 各阶段验收判据可机器判定（§11）。
- [x] 全文不含可直接运行的 Python 实现代码（仅签名与伪代码）。
