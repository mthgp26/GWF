# 复现任务提示词：Gromov-Wasserstein Factorization (GWF)，TPAMI 2022

> **用途**：本文件是后续复现工作的完整任务提示词（验收标准 + 实施计划 + 交付流程）。
> 使用方式：把它作为新会话的首条指令，或在 `code/` 下按 §9 的实施计划逐步推进。
> 所有"必须"项来自论文；标 **[待确认]** 的是原文未明确、需要按 §11 自行决策或核对官方代码的点。

> ## ⛔ 第 0 条硬约束（Highest Priority Rule）
>
> **先把 `code/SPEC.md` 整理成文并定稿，再写任何一行实现代码。**
>
> - 在 `SPEC.md` 通过评审（§9 Phase 1 的门槛）之前，**禁止**创建 `code/gwf/*.py`、`tests/*.py`、`scripts/*.py` 等任何代码文件；数据下载/探索性打印脚本可以写，但不得沉淀为 `gwf/` 包的一部分。
> - 收到"开始复现"类指令时的**第一个动作必须是撰写 `SPEC.md`**，而不是搭脚手架或跑实验。若被要求直接出代码，先指出本条约束并回到 SPEC。
> - `SPEC.md` 是后续所有实现的**唯一技术契约**：代码里每个函数、每个超参都必须能在 `SPEC.md` 的对应条目上找到出处（论文式号/算法号/本文件 §号）。凡 `SPEC.md` 未覆盖的实现细节，**先补 SPEC 再写代码**，不允许"边写边定"。
> - SPEC 定稿后如需变更（例如 §11 的 [待确认] 项被实验锁定），必须**先改 SPEC 并记录变更版本**，再改代码。禁止代码与 SPEC 静默不一致。
> - `SPEC.md` 的详细内容清单见 **§6**，撰写步骤见 **§9 Phase 1**。

---

## 0. 任务陈述（Objective）

在 `code/gwf/` 下用 **PyTorch** 从零实现论文 *"Representing Graphs via Gromov-Wasserstein Factorization"*（Xu, Liu, Luo, Carin；IEEE TPAMI 2022；DOI 10.1109/TPAMI.2022.3153126）中的完整模型，并用真实图数据集验证其复现出来的**图聚类（Rand-Index）**与**图分类（Accuracy）**指标，与论文报告值对照。

**执行顺序（强制）**：`写 SPEC.md` → `评审定稿` → 按 §9 Phase 2 起的顺序实现与验证。

必须实现的三条主线（缺一不可）：

1. **GWF（直推式 / transductive）**：可学习图因子 $U_{1:K}$ + 每图权重 $z_i$（softmax 后为 $\lambda_i$）。
2. **GWD 求解器的两个版本**：`GWD_PPA`（Algorithm 1）与 `GWD_BADMM`（Algorithm 2），二者可插拔互换。
3. **GWF-GNN（归纳式 / inductive）**：GNN 编码器 $h_\theta$ 替代 $z_i$，构成"GNN 编码 + GWF 解码"的自编码框架。

扩展项（分阶段，允许后置）：**FGW / 节点属性版本**、**半监督分类正则**、**交替优化 vs 联合优化**对比。

**参考仓库（仅作交叉核对，不作为抄写来源）**：<https://github.com/SDS-Lab/Gromov-Wasserstein-Factorization>（AAAI-20 + TPAMI-22 官方实现；论文中给出的旧地址为 `HongtengXu/Relational-Factorization-Model`）[待确认：是否能访问，若无法访问则完全以论文公式为准]。

---

## 1. 论文资料位置（先读，再写代码）

| 文件 | 作用 |
| :-- | :-- |
| `paper/GW works/[main] Representing_Graphs_via_Gromov-Wasserstein_Factorization.md` | 论文正文要点（本提示词的公式来源） |
| `paper/GW works/[Intro] Representing_Graphs_via_Gromov-Wasserstein_Factorization.md` | 中文总结，含模型—公式—算法对照速查表（§附录） |
| `paper/GW works/.pdf/abtract.md` | 从**模型结构**（非学习算法）角度对 GWF 的拆解，写代码前先读，用于建立 shape 直觉 |
| `paper/GW works/.pdf/[main] 通过Gromov-Wasserstein因子分解表示图.tex` | 论文 LaTeX 源（公式与算法的**权威版本**，含本提示词中所有 $\gamma$、$H_k$、FGW 细节） |
| `paper/GW works/.pdf/Gromov-Wasserstein Factorization Models for Graph Clustering.pdf` | 前作 AAAI-20，GWD/GWB 模块的原始出处 |

**规则**：凡本提示词与 LaTeX 源冲突，以 **LaTeX 源 / 论文公式**为准，并在 `NOTES.md` 记录冲突。

---

## 2. 环境与工具链

- 工作目录：`/Users/Cr2O7/Documents/我的/南开/大创/research`
- Python：**优先复用仓库已有的 `.venv`**（`.venv/bin/python`，Python 3.14 + torch 2.13，MPS 可用）。
  - 备选：`/Users/Cr2O7/anaconda3/bin/python`（torch 2.11，MPS 可用）。
  - 系统 `/opt/homebrew/bin/python3` **没有 torch**，不要用。
- 依赖：`torch`、`numpy`、`scipy`、`scikit-learn`、`pandas`、`matplotlib`、`tqdm`；数据加载优先 `torch_geometric.datasets.TUDataset`，若不可用则自行解析 TUDataset 原始格式（见 §9.1）。
- 设备：默认 `mps`，若出现非支持算子报错则退到 `cpu`；**必须保证两个设备上结果一致**（在 `NOTES.md` 记录实际使用的设备）。
- 数值：所有 Sinkhorn 类迭代在 **log-space 或带归一化** 下实现，禁止裸 `exp`，见 §13.6。

---

## 3. 记号与张量约定（先固定，避免前后不一致）

| 符号 | 含义 | shape |
| :-- | :-- | :-- |
| $C_i$ | 第 $i$ 个观测图的邻接/距离矩阵 | $(N_i, N_i)$ |
| $\mu_i$ | 观测图节点分布（均匀） | $(N_i,)$，$\mu_i=\frac{1}{N_i}\mathbf{1}$ |
| $U_k=f(V_k)$ | 第 $k$ 个图因子（sigmoid 后） | $(N_k, N_k)$ |
| $\mu_k$ | 图因子节点分布（均匀，固定） | $(N_k,)$ |
| $z_i \in \mathbb{R}^K$ | 图的自由权重参数 | $(K,)$ |
| $\lambda_i=g(z_i)$ | softmax 后的权重（$\Delta^{K-1}$） | $(K,)$，$\sum_k\lambda_{ik}=1$ |
| $T_k$ | $B$（或 $C_i$）到 $U_k$ 的最优传输矩阵 | $(N_b, N_k)$ |
| $B^{(l)}$ | 第 $l$ 步的 GW 重心（重建图） | $(N_b, N_b)$，**$N_b=N_i$，且 $B^{(0)}=C_i$** |
| $H_k$ | 图因子节点的可学习嵌入（仅节点属性版本） | $(N_k, D)$ |
| $C_{XY}$ | GW 二次型的线性项 | $(N_X, N_Y)$ |

**关键约定（务必写进代码 docstring）**：

1. 重心维度与观测图一致：`B.shape == C_i.shape == (N_i, N_i)`；因此 $T_k$ 是 $(N_i, N_k)$，**$T_k$ 的 size 依赖 $i$，不能跨图共享**。
2. 传输多面体：$T_k \ge 0$，$T_k \mathbf{1}_{N_k} = \mu_i$，$T_k^\top \mathbf{1}_{N_i} = \mu_k$。
   （论文中 $N_X$ 扮演 $B$ / 观测图，$N_Y$ 扮演 $U_k$，故列边际为 $\mu_k$。）
3. 图因子尺寸 $N_k$ 原文未给出 **[待确认]**：先默认所有因子同尺寸、且小于观测图（建议 $N_k=10$，对 MUTAG/ENZYMES 这类小图可设 $N_k=\min_i N_i$ 或 $10$），做成 `--factor-nodes` 可配置项；对 $N_k$ 做一次敏感性实验。
4. 索引方向（论文式 21 与 `.pdf/abtract.md` 给出的形式互为转置）**必须用 §4 的等价性测试锁定（`barycenter` 索引方向测试）**，不得凭直觉写死；两种写法只在 $T$ 的行列约定不同时等价。

---

## 4. 数学规格（实现依据）

### 4.1 GW 差异（式 6 / 式 7）

$$
d_{gw}(C_X,C_Y)=\min_{T\in\Pi(\mu_X,\mu_Y)}\Big(\sum_{i,i'}\sum_{j,j'}|c^X_{ii'}-c^Y_{jj'}|^p\,t_{ij}t_{i'j'}\Big)^{1/p}
$$

$p=2$ 时的二次型（**这是代码里真正用的形式**）：

$$
d_{gw}^2(C_X,C_Y)=\min_{T\in\Pi}\ \big\langle C_{XY}-2C_XTC_Y^\top,\ T\big\rangle,\qquad
C_{XY}=(C_X\odot C_X)\mu_X\mathbf{1}_{N_Y}^\top+\mathbf{1}_{N_X}\mu_Y^\top(C_Y\odot C_Y)^\top
$$

（$\odot$ 为逐元素积。注意 §11 的符号问题：在 SPEC 的索引方向判定测试通过前，不要删掉 `-2 C_X T C_Y^T` 的转置选项。）

### 4.2 GWD 模块（Algorithm 1：PPA，$\gamma_{\text{PPA}}=0.01$）

```
输入 C_X(NxNx), C_Y(NyNy), mu_X, mu_Y, M, gamma
C_XY = (C_X*C_X) @ mu_X[:,None] @ ones(1,Ny) + ones(Nx,1) @ mu_Y[None,:] @ (C_Y*C_Y).T
T = mu_X[:,None] @ mu_Y[None,:]      # T^0
a = mu_X
for m in 0..M-1:
    Phi = exp(-(C_XY - 2 C_X @ T @ C_Y.T)/gamma) * T
    b = mu_Y / (Phi.T @ a)           # Sinkhorn-Knopp
    a = mu_X / (Phi @ b)
    T = diag(a) @ Phi @ diag(b)
return T                              # T^(M)
```
- 实现要点：$\exp$ 前对整体做常数平移（`Phi = exp(-(...)/gamma - shift)`，`shift = max(...)`），Sinkhorn 的比例因子本身可吸收该常数；`b`、`a` 的分母加 `eps`。
- 返回 `T` 的同时**建议**（可选）返回 $\langle C_{XY}-2C_XTC_Y^\top,T\rangle^{1/2}$ 作为该图的 GW 差异，用于监控。

### 4.3 GWD 模块（Algorithm 2：BADMM，$\gamma_{\text{BADMM}}=1$）

```
C_XY = 同 4.2
T = mu_X[:,None] @ mu_Y[None,:];  Z = zeros(Nx, Ny)
for m in 0..M-1:
    Phi1 = exp((2 * C_X.T @ T @ C_Y + Z)/gamma) * T
    S    = Phi1 * (mu_Y / (Phi1.T @ ones(Nx)))[None,:]        # 列归一化
    Phi2 = exp(-(C_XY - 2 * C_X @ S @ C_Y.T + Z)/gamma) * S
    T    = (mu_X / (Phi2 @ ones(Ny)))[:,None] * Phi2          # 行归一化
    Z    = Z + gamma * (T - S)
return T
```
- 同样需要 log-space 稳定性处理；`Phi1` 里 `exp(+...)` **必须**做最大值平移，否则 $\gamma=1$ 时仍会上溢。

### 4.4 GW 重心（Algorithm 3：GWB$^{(L)}$，$L=5$）

```
输入 U_{1:K}, lambda, B^0, mu_b, L
for l in 0..L-1:
    for k in 1..K:
        T_k = GWD(B^l, U_k)                        # 注意：B^l 是 X 侧，U_k 是 Y 侧
    B^{l+1} = (1/(mu_b[:,None] @ mu_b[None,:])) * sum_k lambda_k * (T_k @ U_k.T @ T_k.T)
return B^L, T_{1:K}^L
```
- 闭式更新（式 21）来自"给定 $T_k$ 时 $\min_B$"的一步解析解，实现时**不得在对角/归一化上取巧**；$\mu_b\mu_b^\top$ 是外积矩阵，逐元素相除。
- 允许把 $B^{(l)}$ 投影/裁剪到 $[0,1]$（论文中因子与图均非负），但要作为**开关**并在 `NOTES.md` 说明是否影响结果。
- 可选优化：$K$ 个 GWD 调用可用 `vmap` / batch 化（因子同尺寸时把 $U$ 堆成 $(K,N_k,N_k)$，$T$ 堆成 $(K,N_b,N_k)$）。**先在循环版上把数值校验做对，再做向量化优化**，并保证两者输出误差 `< 1e-8`。

### 4.5 损失函数（式 27）

$$
\mathcal{L} = \sum_{i=1}^{I}\|B_i^{(L)}-C_i\|_F^2 \quad\text{（可选：除以 }N_i^2\text{ 归一化，作为开关）}
$$
节点属性版本：$+\ \|F_b^{(L)}-F_i\|_F^2$。$d_{\text{loss}}$ 只用 Frobenius 范数（论文明确说明它比用 GW 差异做 loss 更快且效果相当）。

### 4.6 重参数化（式 12–13）

$$
u^k_{ij}=f(v^k_{ij})=\mathrm{sigmoid}(v^k_{ij}),\qquad \lambda_i=g(z_i)=\mathrm{softmax}(z_i)
$$
- 可学习参数是 **logit 空间**的 $V_{1:K}$ 和 $z_{1:I}$，**不是** $U$ 和 $\lambda$；优化器更新 $V,z$。
- 变量定义域自动满足 $0\le U\le 1$、$\lambda\in\Delta^{K-1}$，因此**不要**再手动加约束/投影（除 §4.4 的重心裁剪）。

### 4.7 包络定理反向传播（Theorem 4.1）

$$
\frac{\partial F(\hat x(\alpha),\alpha)}{\partial\alpha}=\frac{\partial L(\hat x(\alpha),\hat\tau(\alpha),\alpha)}{\partial\alpha}
$$

**实现语义（最容易做错的一点）**：前向结束时拿到 $T_{1:K}^{(L)}$，随后在目标中把它当作**常数**：
$$
\mathcal{L}_i=\Big\|\frac{1}{\mu_b\mu_b^\top}\sum_k g_k(z_i)\,T_k^{(L)}\,f(V_k)\,(T_k^{(L)})^\top-C_i\Big\|_F^2
$$
用 `T_k.detach()` 实现（前向仍要用未 detach 的 $T$ 计算 $B$，只有进入 §4.7 的重建项时视为常数——两者数值相同，但**必须 detach**，否则会沿 Sinkhorn 迭代反传，训练时间会爆炸）。在 `NOTES.md` 中明确写出这一点，并附一段"是否 detach 的耗时对比"实验。
注意：包络定理的适用前提是 $T_k$ 为对应最优解（近似 $M$ 步后成立近似）；文档中标注该近似。

### 4.8 训练算法（Algorithm 4）

```
输入 {C_i, mu_i}, K, 学习率 eta
1. 预热初始化 V_{1:K}: 从数据集中随机取 K 个图 C_(1..K)，V_k = logit(0.98 * C_k + 0.01)
2. z_{1:I} 随机初始化（Normal(0, 小方差)）
3. 因子节点分布取均匀
4. for epoch in 1..E:
5.   for i in 1..I:
6.      B^0 = C_i, mu_b = mu_i
7.      B^L, T_{1:K}^L = GWB^L(f(V), g(z_i), B^0, mu_b)
8.      L_i = ||B^L - C_i||_F^2
9.      固定 T_{1:K}^L，算 ∇_V L_i、∇_{z_i} L_i
10.     Adam 更新 V_{1:K} 与 z_i
```
- 超参（论文值）：$K\in\{15,30,150\}$，$M=20$，$L=5$，$\gamma_{\text{PPA}}=0.01$，$\gamma_{\text{BADMM}}=1$，Adam lr $=0.01$，epochs $=10$，batch = 单图（或小 batch，见 §11）。
- **预热初始化必须实现**：`V_k = logit(0.98*C_k + 0.01)`（对邻接矩阵逐元素做 sigmoid 反函数），它同时保证因子非负且接近真实邻接矩阵，论文称其"一定程度上提升稳定性"。
- **交替优化 vs 联合优化**：两种都要实现（`--optim joint|alternating`）。交替 = 先冻结 $V$ 更新 $z$，再冻结 $z$ 更新 $V$（各自若干 inner step）。论文结论：交替通常聚类效果更好，需复现该对比图（Phase 5 第 5 条）。

### 4.9 归纳式 GWF-GNN（式 15）

$$
\min_{V_{1:K},\theta}\sum_i d^q_{\text{loss}}\big(B(f(V_{1:K}),g(h_\theta(C_i));d_{gw}^{2,(M)}),C_i\big)
$$
- $h_\theta$ 输出维度 $=K$，接 softmax 得到 $\lambda_i$；GNN 作为 encoder、GWF 作为 decoder。
- 无节点属性的图（IMDB-B/M）：输入全 1 向量（论文原文），**[待确认]** 使用节点度作为输入（论文在 IMDB 讨论中提到"GNN 只能用节点度作为输入"，二者取一，用消融确定）。
- 有节点属性且为离散标签时转 one-hot。
- 两种架构：**GIN**（5 层，$K$ 小时更优、带属性图更优）与 **DiffPool**（无属性图有时更优）；层数在 $\{5,10,15\}$ 中选。
- 实现建议：优先用 PyG 的 `GINConv`/`global_mean_pool`；若 PyG 不可用，自行实现 5 层 GIN（`MLP + ε·x + Σ_neighbors`），**不要**因此阻塞主线（GWF 直推式版先跑通）。

### 4.10 扩展 A：节点属性（FGW，式 25–26）

$$
d_{fgw}=\min_{T\in\Pi}\Big(\underbrace{\langle C_{XY}-2C_XTC_Y^\top,T\rangle}_{\text{GW 项}}+\underbrace{\langle D,T\rangle}_{\text{Wasserstein 项}}\Big)^{1/2}
$$
$$
D=(F_X\odot F_X)\mathbf{1}_D\mathbf{1}_{N_Y}^\top+\mathbf{1}_{N_X}\mathbf{1}_D^\top(F_Y\odot F_Y)^\top-2F_XF_Y^\top,\qquad
F_b^{(L)}=\frac{1}{\mu_b\mathbf{1}_D^\top}\sum_{k=1}^{K}g_k(z_i)\,T_k^{(L)}H_k
$$
$$
\mathcal{L}=\|B^{(L)}-C_i\|_F^2+\|F_b^{(L)}-F_i\|_F^2
$$
**实现关键**：FGW 与 GW 的求解器完全相同，只需把 $C_{XY}$ 替换为 $\tilde C_{XY}=C_{XY}+D$（论文明确说明）。因此 `GWD_PPA/BADMM` 应接受一个可选 `cost_extra` 参数；$H_{1:K}$ 作为新增可学习参数（初始化可取自数据集中图的节点特征，或随机）。

### 4.11 扩展 B：半监督（式 24）

$$
\min_{V,z,\phi}\ \sum_i d^q_{\text{loss}}(\cdots,C_i)\ +\ \beta\sum_{i\in\mathcal D_l}L(\phi(z_i),l_i)
$$
即"GWF 表示 + 线性/MLP 分类头"联合训练；$\mathcal D_l$ 为有标签子集，$\beta$ 可调。

---

## 5. 从论文中抽取的复现目标数值（验收对照表）

### 5.1 图聚类（Rand-Index %，论文表 7.3 摘录）

| 数据集 | GWF$_{\text{PPA}}$ | GWF$_{\text{BADMM}}$ | GWF-GNN |
| :-- | :-- | :-- | :-- |
| AIDS | 99.0±0.4 | 98.4±0.5 | 99.0±0.6 |
| PROTEIN | 59.8±0.2 | 60.0±0.3 | 61.0±0.3 |
| ENZYMES | 72.3±0.3 | 71.9±0.5 | 72.9±1.1 |
| MUTAG | 69.2±2.0 | 68.8±2.4 | 69.4±2.3 |
| PTC-MR | 52.5±1.1 | 52.4±1.8 | 52.8±1.6 |
| IMDB-B | 51.8±0.7 | 51.3±0.5 | 51.2±1.2 |
| IMDB-M | 55.9±0.4 | 55.1±0.6 | 56.2±0.9 |

### 5.2 图分类（Accuracy %，论文表 7.4 摘录）

| 方法 | PROTEINS-F | ENZYMES | MUTAG | PTC-MR | IMDB-B | IMDB-M |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| GWF$_{\text{PPA}}$(K=30) | 73.89±2.05 | 73.06±3.96 | 82.25±3.67 | 62.27±2.53 | 60.93±2.68 | 39.97±1.35 |
| GWF$_{\text{BADMM}}$(K=30) | 73.33±2.41 | 72.76±4.14 | 82.41±3.02 | 61.22±2.40 | 60.42±3.81 | 40.28±1.07 |
| GWF-GNN$_{\text{DiffPool}}$(K=150) | 73.68±2.44 | 73.03±4.08 | 86.91±2.10 | 61.34±1.91 | 72.90±0.60 | 48.10±0.31 |
| GWF-GNN$_{\text{GIN}}$(K=150) | 74.31±3.02 | 73.80±4.16 | 90.90±0.48 | 62.46±1.06 | 69.70±1.10 | 48.04±1.41 |

### 5.3 数据集统计（论文表 7.1）

| 数据集 | #graphs | #classes | 节点属性维度 | 平均节点数 | 平均边数 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| AIDS | 2000 | 2 | 4 | 15.69 | 16.20 |
| PROTEIN | 1113 | 2 | 1 | 39.06 | 72.82 |
| PROTEIN-F | 1113 | 2 | 29 | 39.06 | 72.82 |
| ENZYMES | 600 | 6 | 18 | 32.63 | 62.14 |
| MUTAG | 188 | 2 | 7 | 17.93 | 19.79 |
| PTC-MR | 344 | 2 | 18 | 14.29 | 14.69 |
| IMDB-B | 1000 | 2 | — | 19.77 | 96.53 |
| IMDB-M | 1500 | 3 | — | 13.00 | 65.94 |

**验收容差（工程判定，写进报告）**：分类/聚类指标与论文值的偏差在 $\pm 3$ 个百分点内视为"复现成功"；表格中同时给出"是否落在论文 ± 标准差范围内"。

### 5.4 算法对比结论（需要在 Phase 3 第 7 条实验中复现）

- 无向无权图：**PPA 收敛更快**；带权图：**BADMM 在足够迭代后 GW 差异更低**。
- 两者在精度和收敛性上均**优于**基于熵正则化的 GW 方法（论文 Fig. comparison，对比对象为 Peyré et al. ICML 2016 / Xu et al. NeurIPS 2019）。
- 复杂度：单次 GWD 迭代 $O(NE)$（利用稀疏性，稠密实现是 $O(N^3)$）；GWF 训练 $O(BKNE)$；**测试阶段 GWF 仍为 $O(BKNE)$，GWF-GNN 仅 $O(BK)$**。

---

## 6. 编码前置交付物：`code/SPEC.md`（**必须先于代码完成**）

`SPEC.md` 是本次复现的**技术契约与唯一真相来源**，目标是把论文（含 §11 的歧义）翻译成"照着就能实现、且能被别人评审"的精确规格。**它必须在写任何实现代码之前完成并评审通过。**

### 6.1 撰写原则（这几条决定 SPEC 的质量）

1. **可执行性优先于可读性**：每条规格都写成"输入 shape/类型 → 计算步骤 → 输出 shape/类型 → 数值容差"。凡是能用断言表达的，就写成断言。
2. **无歧义**：禁止出现"大致""适当""类似论文"这类措辞。$\gamma$、$M$、$L$、$K$、$N_k$、lr、epochs 必须是**具体数值 + 默认值 + 取值范围**。
3. **可追溯**：每个规格条目必须标注出处，格式 `[式 21]`、`[Alg.1]`、`[§11-3]`（本提示词条目）、`[决策]`（自主决定并说明依据）。无法追溯到论文的，必须标 `[决策]` 或 `[待确认]`。
4. **歧义显式化**：§11 的 10 条 [待确认] 在 SPEC 中必须**逐条给出"默认取值 + 判定实验 + 判定标准"**，不允许留白。特别是索引方向（$C_{XY}$ 转置、$T_kU_k^\top T_k^\top$ vs $T_k^\top U_k T_k$）必须写明由哪个测试锁定。
5. **只定规格、不写实现**：可以写张量形状、数学步骤、伪代码、公式编号，但**不写完整可运行的 Python 函数体**。伪代码允许（描述算法流程），代码不允许。
6. **版本化**：文档头部记 `版本 / 日期 / 依据的论文文件路径与 git commit`；每次因实验结论修改 SPEC，追加 Changelog 条目并说明改动原因，禁止静默覆盖。

### 6.2 `SPEC.md` 必含章节（缺一节不得开写代码）

| # | 章节 | 必须写清的内容 |
| :-- | :-- | :-- |
| 1 | 目标与范围 | 复现哪三条主线、哪些扩展后置、明确不做什么（out of scope） |
| 2 | 记号与张量契约 | 全部符号、shape、dtype、device；$N_b=N_i$、$B^{(0)}=C_i$、$T_k$ 尺寸随图变化等硬约定 |
| 3 | 数据规格 | 数据集清单与统计量、解析规则、邻接对称化/归一化/对角处理、属性与 one-hot 规则、无属性图的特征构造、`Graph` 数据结构定义 |
| 4 | 核心算子规格 | `C_XY` 构造、GWD_PPA、GWD_BADMM、GWB：**入参/出参签名 + 逐步伪代码 + 数值稳定策略 + 迭代终止条件 + 容差** |
| 5 | 模型规格 | 可学习参数清单（$V_{1:K},z_{1:I},H_{1:K},\theta$）及 shape、重参数化映射、前向计算图、什么量参与梯度、**哪些张量必须 detach 及理由** |
| 6 | 目标函数与梯度 | 损失定义、是否归一化、包络定理的落地方式、梯度检查方案（`gradcheck` 的配置与容差） |
| 7 | 训练规格 | 初始化（含预热 `logit(0.98C+0.01)`）、优化器/超参、epoch/batch、**每轮内的更新顺序**、joint vs alternating 的精确定义（含 inner step 数） |
| 8 | 评估规格 | K-means 配置与 RI 口径、SVM 10 折 CV 与 $C$ 搜索、**std 的统计口径**、随机种子策略、结果 JSON 字段定义 |
| 9 | 歧义登记表 | §11 十条逐条落实：默认值 / 判定实验 / 判定标准 / 状态（未定\|已定+依据） |
| 10 | 溯源表（论文 ↔ SPEC ↔ 代码 ↔ 测试） | 表格：`论文式号/Alg.号 → SPEC 条目号 → 计划函数名 → 验证该条目的测试 id → 当前状态`。此表是"SPEC 与代码一致"的核查凭据，Phase 9 交付时须全部填满 |
| 11 | 验收标准 | 各阶段验证门槛的可执行判据（含 §5 的容差）、必过的测试清单 |
| 12 | 接口清单 | 计划实现的模块/函数签名与依赖关系（用于评审，不是实现） |

### 6.3 评审与冻结流程（§9 Phase 1 的门槛）

```
撰写 SPEC.md  →  自检 §6.2 十二节齐全 + §6.4 检查清单全过
              →  提交评审（用户/评审方确认）
              →  冻结 SPEC v1.0（记版本与日期）
              →  此后才允许创建 code/gwf/*.py
```
- 评审未通过前，**不得**开始 Phase 2。若评审提出修改，改 SPEC 后重新提交，直至通过。
- 评审通过后如需变更 SPEC：先改 SPEC + 记 Changelog → 再改代码；`REPORT.md` 中需列出 SPEC 的全部变更历史及其对结果的影响。

### 6.4 SPEC 定稿自检清单（逐条打勾后才能提交评审）

- [ ] 十二条章节齐全，无空章节。
- [ ] 所有超参都有具体默认数值，没有"待定"。
- [ ] 每个函数签名都标了输入/输出的 shape 与 dtype。
- [ ] 每个公式条目都标了论文出处或 `[决策]`。
- [ ] §11 十条 [待确认] 全部有结论或明确的判定实验设计。
- [ ] 索引方向问题的判定测试已写明断言内容与容差。
- [ ] detach 规则、数值稳定策略（log-space/平移）、eps 位置已写明。
- [ ] 数据统计校验的期望值（§5.3）已写入。
- [ ] 各阶段验收判据可被机器判定（不是"效果良好"这类主观描述）。
- [ ] 全文不含可直接运行的 Python 实现代码（伪代码除外）。

---

## 7. 训练/评估流程规范

**图聚类（§7.3 协议）**：train GWF（或 GWF-GNN）→ 取 $z_{1:I}$（$g(z_i)$ 或 softmax 后的 GNN 输出，**统一用 $\lambda_i$ 还是 $z_i$ 需固定，建议 $\lambda_i$**）→ 直接 K-means（簇数 = 类别数）→ 计算 Rand-Index。

**图分类（§7.4 协议）**：用学到的表示（$K=30$ 或 150）→ 10 折交叉验证 → **SVM** 分类（LIBSVM 风格；$C$ 从 $\{10^{-3},10^{-2},\dots,10^{3}\}$ 中在训练折上交叉验证选取）→ 报告 accuracy。

**随机性**：每个配置至少 5 个随机种子（论文报 std，多次运行的均值±std）；固定种子的记录写入结果 JSON；所有超参、设备、耗时都要落盘。

**运行日志规范**：每次实验输出 `results/<dataset>/<model>_K<K>_<optim>_seed<S>.json`，字段至少包含：`dataset, model, K, M, L, gamma, lr, epochs, optim, seed, metric, value, runtime_sec, device, git_commit`。

---

## 8. 数据

### 7.1 加载
- 首选 `torch_geometric.datasets.TUDataset(root='data/', name='MUTAG')`；若无 PyG，则按 TUDataset 原始格式自行解析（`*_A.txt` 邻接、`*_graph_indicator.txt`、`*_graph_labels.txt`、`*_node_labels.txt`、`*_node_attributes.txt`），写出 `gwf/data.py`，接口统一：
  ```python
  load_dataset(name) -> list[Graph], torch.LongTensor  # Graph: C (Ni,Ni) float, F (Ni,D) float or None, y int
  ```
- 下载地址：<https://chrsmrrs.github.io/datasets/>。
- 必须写一个 `check_dataset_stats()`，把解析结果与 §5.3 表格逐列比对（#graphs/#classes/平均节点数/平均边数），**不完全一致就先修数据加载，不要开始训练**。

### 7.2 预处理
- 邻接矩阵：对称化（可选开关）、归一化到 $[0,1]$（论文中 $C\in[0,1]^{N\times N}$），对角置 0。
- 无节点属性的图（IMDB-B/M）：构造"全 1"或"度"特征作为 GNN 输入的两种变体（消融）。
- 离散节点标签（MUTAG 等）：one-hot。
- 图因子尺寸 $N_k$、是否使用边权：见 §11。

---

## 9. 分阶段实施计划与验证门槛（**每阶段必须通过验证才进入下一阶段**）

### Phase 1：撰写并冻结 `code/SPEC.md`（**编码前置门槛**）
按 §6 的要求撰写 `code/SPEC.md`（十二条章节齐全，见 §6.2），逐条通过 §6.4 自检清单后提交评审。
1. **禁止**在本阶段创建任何实现代码文件（`gwf/*.py`、`tests/*.py`、`scripts/*.py`）。
2. 本阶段允许的产出：`SPEC.md`，以及对论文/数据集的只读核对记录（写入 `NOTES.md`）。
3. **通过标准**：① §6.2 十二节齐全且无空章节；② §6.4 自检清单全部打勾；③ §11 的十条 [待确认] 全部给出"默认值 + 判定实验 + 判定标准"；④ 溯源表（SPEC 第 10 节）覆盖 §4 的全部公式/算法；⑤ 评审方确认通过。
4. **冻结**：记为 `SPEC v1.0`（含日期与 git commit），此后才允许进入 Phase 2。
5. 若评审未通过：修改后重新提交，直到通过；不得以"边实现边补 SPEC"的方式绕过。

### Phase 2：脚手架 + 数据（按 SPEC 实现）
1. 建仓库结构（§10），写 `README.md`、`requirements.txt`、`configs/*.yaml`。
2. 实现数据加载 + `check_dataset_stats()`。
3. **通过标准**：MUTAG/PROTEIN/ENZYMES 统计量与 §5.3 一致（±0.5%）。
4. 产出：`tests/test_data.py` 全绿。

### Phase 3：GWD 求解器（PPA + BADMM）**独立验证**
实现 `gwf/transport.py`，先不接模型。写 `tests/test_gwd.py`：

1. **边际约束**：随机图对（$N_X,N_Y\in[5,30]$），PPA/BADMM 各 $M\in\{1,5,20\}$，断言
   `T >= -1e-12`、`|T.sum(1) - mu_X| < 1e-6`、`|T.sum(0) - mu_Y| < 1e-6`。
2. **梯度不存在**：断言 `T.requires_grad is False`（求解器内部不建图）。
3. **对称/恒等**：$C_X=C_Y$ 且 $\mu_X=\mu_Y$ 时，$d_{gw}^2=0$ 应可达到（对权图）；无向无权图上 $d_{gw}$ 不为 0（辨识问题，论文 §6.1 结论 3）——断言"权图 $d_{gw}$ 明显小于无权图"。
4. **置换等变性**：对 $C_Y\leftarrow P C_Y P^\top$，断言 $T \leftarrow T P^\top$，且 $d_{gw}$ 不变（容差 `1e-6`）。（这是"置换不变表示"的基础，必须过。）
5. **优化单调性**：$d_{gw}$ 随 $M$ 增加**单调不增**（允许 `1e-8` 抖动）；PPA 与 BADMM 在 $M=200$ 时差异 `< 5%`（权图上 BADMM 更低）。
6. **退化情形**：$N_X=1$ 或 $N_Y=1$；全零图；常数图 → 不产生 NaN/Inf。
7. **难度对照（可选但建议）**：BA 模型生成 100 对 50 节点同构图（无权/有权、有向/无向 4 类），画"GW 差异 + 运行时间 vs 迭代步数"曲线，复现论文 Fig. comparison 的定性结论（§5.4），并与熵正则化基线（`scipy` 或自实现 Sinkhorn-GW）对比。

**通过标准**：以上 1–6 全绿；7 的定性趋势与论文一致。产出 `NOTES.md` 记录 $\gamma$ 与数值稳定性的实际处理。

### Phase 4：GWB 重心模块
实现 `gwf/barycenter.py`。测试：
1. **形状**：`B.shape == (N_b, N_b)`，`len(T_list) == K`。
2. **单因子恒等**：$K=1$、$\lambda=[1]$、$U_1=C$（同尺寸 $N_k=N_b$）时，$B$ 应收敛到与 $U_1$ 同构（在节点置换意义下）：断言 `torch.cdist/sorted` 结构量近似相等，或直接用 $d_{gw}(\hat B, U_1) < \epsilon$。
3. **凸组合极限**：两个同尺寸因子、$\lambda=(1,0)$ 时 $B\approx U_1$；$\lambda=(0.5,0.5)$ 时 $B\approx$ 两者的 GW 平均（用 $d_{gw}(B,U_1)\approx d_{gw}(B,U_2)$ 断言）。
4. **$L$ 收敛**：$L\in\{1,5,20,50\}$ 时 $B^{(L)}$ 的变化量随 $L$ 递减。
5. **索引方向锁定（§3 第 4 条）**：分别用
   (a) $B=\frac{1}{\mu_b\mu_b^\top}\sum_k\lambda_k T_k U_k^\top T_k^\top$（式 21）
   (b) $B=\frac{1}{\mu_b\mu_b^\top}\sum_k\lambda_k T_k^\top U_k T_k$（`.pdf/abtract.md` 形式）
   两种实现，在**同尺寸**（$N_k=N_b$）恒等测试上判定哪种能通过测试 2；把结论写进 `NOTES.md` 并在代码注释里引用该测试。（若两者在 $N_k=N_b$ 时都通过，则改用 $N_k\ne N_b$ + 前向损失更低的那个，并用 $d_{gw}(B,C_i)$ 作为判据。）
6. **接口**：`gwb(U_list, lam, B0, mu_b, L, gwd='ppa'|'badmm', **kw) -> (B, T_list)`。

### Phase 5：GWF 直推式模型端到端
实现 `gwf/model.py`（`GWF` 模块）+ `gwf/train.py`。
1. **冒烟测试**：合成数据（$I=20$，全部 12 节点，随机块模型生成 3 类）→ 训练 30 epoch → 重建误差下降 ≥ 50%，K-means RI > 0.9（合成数据应收敛得好）。
2. **梯度正确性**：用 `torch.autograd.gradcheck`（double 精度、$N_k=4$、$M=3$、$L=2$、$K=2$、$I=2$）验证 $\partial\mathcal L/\partial V$ 与数值梯度一致（容差 `1e-4`）；同时断言 **detach 后梯度与未 detach 时的梯度在数值上近似一致但更快**（§4.7），把耗时对比写入 `NOTES.md`。
3. **过拟合测试**：$I=5$、$K=15$、epochs=200 → 训练损失应接近 0。
4. **重参数化有效性**：断言训练后 `U = sigmoid(V) ∈ [0,1]`、`lam = softmax(z)` 行和 = 1（容差 `1e-6`）。
5. **交替 vs 联合**：在 MUTAG（$K=15$，seed 0）上比较两种策略的 loss 曲线与 RI，输出对比图（复现论文 §7.3 的 Fig. alternating 结论方向）。
6. **通过标准**：MUTAG $K=15$、PPA、10 epoch 的 RI 与论文 69.2 的偏差 ≤ 3（若偏差大，先查 §11（论文未明确项）与 §12（已知坑）的坑，再考虑是不是论文报告的 $K$/epoch 配置不同）。

### Phase 6：真实数据集复现聚类表
1. 在 AIDS / PROTEIN / ENZYMES / MUTAG / PTC-MR / IMDB-B / IMDB-M 上跑 GWF$_{\text{PPA}}$、GWF$_{\text{BADMM}}$，$K\in\{15,30\}$，5 seeds。
2. 汇总成与 §5.1 同格式的 markdown 表（含均值±std、与论文差值列）。
3. **通过标准**：大多数数据集落在 ±3 个百分点内；**若某些数据集系统性偏低，必须在报告中明确指出并给出原因假设**（不允许静默降低标准）。

### Phase 7：GWF-GNN（归纳式）
1. 实现 GNN encoder（GIN 优先，DiffPool 次之），输出维度 $K$。
2. 联合训练 $V_{1:K}$ 与 $\theta$（Algorithm 4 的修改版）。
3. 验证：
   - **归纳推理正确性**：对训图直接前向 vs 重新优化 $z_{\text{new}}$（式 `eq:gwf_new`），二者得到的 $\lambda$ 应接近（容差放宽，作为 sanity check）。
   - **效率对照**：测试阶段 GWF-GNN 每图耗时 $\ll$ GWF（论文 $O(BK)$ vs $O(BKNE)$），在 AIDS（2000 图，$K=30$）上测"每 epoch 运行时间 + 每 batch 内存"，复现论文表 complexity 的量级关系（GNN 训练与 GWF 相当，测试显著更快）。
   - **聚类表**：填 §5.1 的 GWF-GNN 列。
   - **分类表**：10 折 CV + SVM 复现 §5.2 的 GWF-GNN 行（$K=150$，GIN/DiffPool）。
4. **通过标准**：GWF-GNN 至少在一个数据集上优于直推式 GWF；分类表偏差 ≤ 3 个百分点。

### Phase 8：扩展项（按优先级）
1. **FGW / 节点属性**（PROTEIN-F、ENZYMES）：实现 $\tilde C_{XY}=C_{XY}+D$、$H_{1:K}$、$F_b$ 与组合损失；验证：加入属性后 ENZYMES/PROTEINS-F 的分类/聚类**不劣于**无属性版本。
2. **半监督**（式 24）：在 MUTAG/PTC-MR 上扫描标签比例 $\{10\%,25\%,50\%\}$ 与 $\beta$，报告曲线。
3. **$N_k$、$K$、$M$、$L$ 敏感性**：$K$ 增大时性能的变化趋势（论文：GWF-GNN 随 $K$ 提升，**IMDB 例外**）——复现该趋势。
4. **可视化**：t-SNE（表示）+ 图因子邻接矩阵热力图（应呈稀疏结构，且与某些样本图相似）——复现论文 Fig. tsne / Fig. factors 的定性结论。

### Phase 9：交付物
- `code/SPEC.md`：**首要交付物**（Phase 1 冻结版 + 全部变更历史），必须与最终代码一致，不得停留在过时版本。
- `code/gwf/` 完整可运行代码 + `README.md`（含一键复现命令）。
- `results/` 结果 JSON + 汇总表 markdown。
- `figures/`：loss 曲线、交替 vs 联合、GW 差异 vs 迭代、t-SNE、图因子热力图、运行时间/内存对比。
- `REPORT.md`：**复现报告**，必须包含：论文值 vs 复现值对照表（含差值）、未复现/偏差较大的项及原因分析、[待确认] 项的最终选择与依据、SPEC 变更史及其对结果的影响、复杂度实测数据、复现难度与工程经验总结。
- `NOTES.md`：所有实现决策、歧义记录、踩坑记录。

---

## 10. 目标仓库结构

```
code/
├── prompt.md                 # 本文件（任务提示词）
├── SPEC.md                   # ★ 技术契约，Phase 1 定稿，先于一切代码
├── REPORT.md                 # 复现报告（Phase 9 产出）
├── NOTES.md                  # 决策与踩坑记录（全程维护）
├── requirements.txt
├── configs/
│   ├── gwf_mutag.yaml
│   ├── gwf_aids.yaml
│   └── gwf_gnn_enzymes.yaml
├── gwf/
│   ├── __init__.py
│   ├── data.py               # TUDataset 加载、统计校验、属性/度特征
│   ├── transport.py          # GWD_PPA, GWD_BADMM, 熵正则化基线, gw_discrepancy()
│   ├── barycenter.py         # GWB(L)
│   ├── model.py              # GWF (V, z, U, lambda), FGW 版本, GNN encoder
│   ├── train.py              # Algorithm 4（joint / alternating），envelope detach
│   ├── evaluate.py           # K-means + RI；SVM 10 折 CV
│   └── utils.py              # 种子、日志、计时、可视化
├── tests/                    # test_data.py, test_gwd.py, test_gwb.py, test_grad.py
├── scripts/                  # run_clustering.py, run_classification.py, run_sensitivity.py
└── results/ , figures/
```

**创建顺序约束**：`SPEC.md` 必须先于 `gwf/`、`tests/`、`scripts/` 三个目录存在；`├── gwf/` 及以下所有条目只能在 SPEC 评审通过后创建。

**代码规范**：每个函数 docstring 里标注对应的论文公式/算法编号；$\gamma$、$M$、$L$、$K$、$N_k$、`optim` 全部走 config，不写死在函数里；所有随机性由统一 `set_seed()` 控制；代码中出现的每个超参必须能在 `SPEC.md` 找到对应条目（PR/提交说明中需注明对应 SPEC 小节）。

---

## 11. 论文中未明确 / 需要决策的点（每条都要给结论 + 证据）

1. **$C_{XY}$ 的转置约定**与 **$T_k U_k^\top T_k^\top$ vs $T_k^\top U_k T_k$**（§3 第 4 条、§11 第 1 条）——**最高优先级**，用同尺寸恒等测试锁定。
2. **图因子尺寸 $N_k$**：原文只说要"预先设定"，未给数值。默认 $N_k=10$；做 $N_k\in\{5,10,15,20\}$ 敏感性。
3. **训练是逐图更新还是小 batch**：Algorithm 4 是 per-graph 循环；论文 $O(BKNE)$ 又暗示有 batch。默认先 per-graph（最忠实），再实现小 batch 版本并比较（$B=20$ 与论文复杂度实验一致）。
4. **交替优化的 inner step 数与顺序**：原文只说"交替"更好。默认每轮：update $z$（1 step）→ update $V$（1 step），做 1/3/5 step 的消融。
5. **IMDB 无属性图的 GNN 输入**：全 1 还是度？（论文两处表述不同）→ 消融决定。
6. **$z_{1:I}$ 的随机初始化分布与尺度**：默认 `N(0, 0.01)`；softmax 后的初始 $\lambda$ 应接近均匀，检查之。
7. **损失是否按 $N_i^2$ 归一化**：默认原地用 $\|\cdot\|_F^2$，但大图上不稳定，做开关。
8. **Sinkhorn 温度/平移的等价性**：确认平移不改变 $T$（比例因子可吸收），在 `NOTES.md` 给出证明式说明。
9. **K-means 使用的表示**：$\lambda_i$（softmax 后）还是 $z_i$；默认 $\lambda_i$（有界、和为 1），并做对照。
10. **PPA 的 $\gamma=0.01$ 数值风险**：$1/\gamma=100$，$\exp$ 极易上/下溢 → 必须 log-space/平移归一化，并把"是否出现 NaN"作为断言写进测试。

---

## 12. 已知坑与防错清单（务必逐条在代码/测试中体现）

1. **$C_{XY}$ 的两个外积项写反**（$\mu_X\mathbf{1}^\top$ 与 $\mathbf{1}\mu_Y^\top$ 对应哪个矩阵）→ 用"$C_X=C_Y$ 时 $d_{gw}=0$ 且 $T=\mathrm{diag}(\mu)$"这条测试兜底。
2. **$C_{XY}$ 依赖 $p=2$**：$(C\odot C)$ 只对 $p=2$ 成立；若改 $p$ 需重推，代码里断言 $p=2$。
3. **忘记 detach $T$** → 反向传播穿进 Sinkhorn 迭代，显存/时间爆炸；写成必需断言（`assert not T_k.requires_grad`）。
4. **$\mu_b\mu_b^\top$ 除法方向错**：它是 $(N_b,N_b)$ 的外积矩阵，逐元素除；注意 `B * (1/outer)` 而非矩阵逆。
5. **$T_k$ 尺寸随图变化**：不能预分配一个全局 $T$ 张量；注意 device/dtype 一致性（float32 vs float64 在 `1e-6` 断言上会失败）。
6. **裸 `exp` 上溢/下溢**（尤其 BADMM 的 $\exp(+(\cdot)/\gamma)$）→ 统一 `logsumexp`/最大值平移；断言中间量 `isfinite`。
7. **均匀分布假设**：$\mu$ 全为 $1/N$；一旦做带权节点分布，"列边际 $=\mu_k$"必须同步改，否则边际测试会静默失败。
8. **K-means 的随机性**：用固定 seed + `n_init>=10`，否则 RI 波动会掩盖真实差异。
9. **SVM 的 $C$ 泄漏**：超参只能在训练折内交叉验证选择（论文明确说明），不要在测试折上调。
10. **性能陷阱**：per-graph 双重循环（$I\times L\times K\times M$）在 AIDS（$I=2000$）上极慢——先保证正确性，再用（a）$K$ 个 GWD 批量化、（b）`torch.compile`/`vmap`、（c）float32 + MPS 加速；记录每次优化的加速比与数值差异（`< 1e-6`）。
11. **论文的 std 是多次运行还是交叉验证折间**：分类表是 10 折 CV 的 std；聚类表是多次运行的 std。报告口径要与论文一致，并在表中注明。
12. **不要把"没复现出来"直接改成"论文有问题"**：先做完 §11 的消融找原因，报告里给出证据链。

---

## 13. 完成定义（Definition of Done）

- [ ] **`SPEC.md` 已在 Phase 1 评审通过并冻结**，且最终版本与代码实现一致（无静默不一致项）。
- [ ] §9 Phase 1–7 全部通过各自验证门槛，测试全绿（`pytest -q`）。
- [ ] 复现出与论文同格式的**聚类表**（7 数据集 × 3 模型）与**分类表**（6 数据集 × 4 行），含偏差列。
- [ ] 复现出 PPA vs BADMM 的定性结论（无权图 PPA 快、权图 BADMM 差异低）与复杂度量级关系。
- [ ] 交替 vs 联合优化的对比结论与论文一致（附曲线图）。
- [ ] 至少完成 §9 Phase 8 的 FGW/节点属性扩展。
- [ ] `REPORT.md` 包含完整的"论文值 vs 复现值"对照、偏差原因分析、[待确认] 项结论。
- [ ] `NOTES.md` 记录全部实现决策与踩坑过程，代码中每处公式都有论文编号引用。
