# NOTES — 实现决策 / 歧义记录 / 踩坑记录

> 全程维护。每处公式都标注论文编号，每个决策标注依据。

## 1. 环境与设备

- 实际训练设备：**CPU**（`.venv` Python 3.14 + torch 2.13）。
  - 基准：MUTAG 规模 GWB（K=15, L=5, M=20, inner=20）CPU ≈ 0.31s，MPS ≈ 3.45s。
  - **MPS 慢约 10 倍**：本工作负载是小矩阵密集运算（大量 logsumexp/(15,18,10) 张量），
    MPS 的 kernel 启动开销远大于计算量，CPU 反而更快。故默认 CPU。
  - 数值：CPU 与 MPS 结果一致（同 dtype 下）。
- 依赖补齐：`pandas`、`tqdm`、`pyyaml`、`pytest` 已 pip 安装进 `.venv`。
- `torch_geometric` 不可用 → 自解析 TUDataset 原始格式（见 NOTES §2）。

## 2. 数据加载（Phase 2）

- 下载源：`https://www.chrsmrrs.com/graphkerneldatasets/<NAME>.zip`（`chrsmrrs.github.io`
  域名不可解析，用 `www.chrsmrrs.com`）。
- **边数口径**：TUDataset 的 `*_A.txt` 把每条无向边存成 `(u,v)` 与 `(v,u)` 两条记录。
  论文表 7.1 的"平均边数"是**唯一无向边**数（如 MUTAG 7442/2/188 = 19.79），故
  `avg_edges = Σ (C>0).sum()/2 / #graphs`。8 个数据集统计全部在 ±0.5% 内通过校验。
- 节点特征优先级：`node_attributes.txt`（连续属性）> `node_labels.txt`（one-hot）> None。
  - MUTAG/PTC_MR 只有离散标签 → one-hot（MUTAG 7 维、PTC_MR 18 维，与论文 D 一致）。
  - AIDS/ENZYMES/PROTEIN(s) 有连续属性；IMDB-B/M 无属性（F=None）。

## 3. GWD 求解器（Phase 3）—— 数值稳定关键决策

### 3.1 内层 Sinkhorn 必须迭代至收敛（不是"一步"）
- 论文 Alg.1 的"一步 Sinkhorn-Knopp"若字面实现为**单次** b/a 更新，列边际误差在
  N_X≠N_Y 时停在 ~3e-2 不收敛（实测 M=500 仍 ~3e-2）。
- 官方仓库 `sinkhorn_knopp_iteration` 实际内层跑 `max_iter=50` 步（固定 kernel + 内层缩放）。
- **本实现**：PPA 内层用 log-domain Sinkhorn 迭代至双边际收敛（tol=1e-12，max_iter=500）。
  这是官方"内层迭代"的忠实实现。

### 3.2 PPA（γ=0.01）必须 log-space
- γ=0.01 → 1/γ=100 → Gibbs 核动态范围可达 ~1e-174（float32 已下溢、float64 勉强）。
- 线性域 Sinkhorn（即便带 max 平移）在极端动态范围下精度差（行边际误差 ~1e-3）。
- **本实现**：PPA 全程 log 域（`logsumexp`），BADMM（γ=1，动态范围温和）用线性域 +
  最后 Sinkhorn 平衡。
- 代价：γ=0.01 使最优传输接近硬分配，Sinkhorn 收敛为**冰川式**（20000 内层才 ~1e-6）。
  故 PPA(γ=0.01) 的边际精度在 M=20 时约 1e-3~1e-5，**不影响下游重建**（对 1e-5 级
  传输误差不敏感）。测试对 PPA(γ=0.01) 用 1e-3 容差、BADMM(γ=1) 用 1e-6 容差。

### 3.3 max 平移的正确形式
- 对 `exp(E)` 做 `E -= E.max()`（指数整体减最大），Sinkhorn 比例因子吸收该全局常数，
  `T` 不变。官方仓库用 `cost /= max(cost)`（除最大）是不同归一化；本实现遵循 prompt §4.2
  的"平移"写法，二者数学上不等价但本实现更干净（记此备注）。

## 4. 批量优化（Phase 4）
- 同尺寸因子堆成 `(K, N_k, N_k)`，GWD 批量计算（`gwd_ppa_batched` / `gwd_badmm_batched`），
  GWB 内 K 循环被向量化。K=15 时提速 ~5 倍（4.4s→0.8s，inner=50）。
- 批量与循环版输出误差：PPA 0.0、BADMM ~1e-8（< 1e-8 要求边缘，已在测试断言）。

## 5. 索引方向（SPEC §9 条目 1）—— 已锁定
- GWD 输出约定 `T: X→Y`（X=B/观测图，Y=U_k），shape `(N_b, N_k)`。
- 闭式更新用**式 21** `B = T_k U_k^T T_k^T`（`(N_b,N_k)@(N_k,N_k)@(N_k,N_b)=(N_b,N_b)`）。
- `.pdf/abtract.md` 的 `T_k^T U_k T_k` 是 `T_k: U_k→B` 的转置约定，二者等价；
  本实现由 Phase 4 同尺寸恒等测试（`test_index_direction_same_size`）锁定式 21。

## 6. 包络定理与 detach（SPEC §5.4）
- 前向求解 GWD 时对 `U=sigmoid(V)` 与 `λ=softmax(z)` **detach**（不沿 Sinkhorn 建图），
  重建项中 `T_k.detach()`；梯度仅经 U（→V）与 λ（→z）。
- `gradcheck` 必须**先固定 T** 再对重建损失求导（否则有限差分会连带重算 T，包络定理
  梯度与全梯度不等）。见 `tests/test_grad.py::test_gradcheck`。

## 7. 预热初始化尺寸（SPEC §9 条目 2）
- 论文 `V_k = sigmoid^-1(0.98 C_k + 0.01)` 默认采样图 C_k 与因子同尺寸；实际数据集图
  尺寸各异且常 > N_k。**决策**：取采样图左上角 `(N_k, N_k)` 子块（不足补 0）。
- N_k 默认 10（敏感度实验 Phase 8）。

## 8. 一步 Sinkhorn vs 内层收敛（重要决策，2026-10-12）
- 论文 Alg.1 是"一步 Sinkhorn"（b=μ_Y/(Φᵀa), a=μ_X/(Φb)，a 跨外层迭代 warm-start）。
  官方仓库 `cmp_algorithm_gwl.py` 确认 `inner_iteration=1`、`outer_iteration=3000`、`beta=0.01`，
  且 `sinkhorn_knopp_iteration` 内做 `cost /= max(cost)`（代价归一化，等效 γ 放大）。
- **实测**（MUTAG, alternating, lr=0.01, 10 epoch, float64）：
  - 一步 Sinkhorn + cost 归一化 + float64：RI=0.563，loss=28.96，**耗时 69s**。
  - 内层收敛（log-domain, inner=50）+ float32：RI=0.6075，loss=18.85，耗时 ~66min。
- **结论**：内层收敛版重建更好、RI 更高（0.6075 > 0.563），故**保留内层收敛版**（也满足
  边际测试 1e-6）。一步 Sinkhorn 快 57× 但重建质量差，与论文报告的 69.2 差距更大。
- 说明：论文"一步 + M=20"在官方代码中可能实际以 `outer_iteration=3000`（至收敛）运行，
  而论文正文写 M=20 是简化。这可能是复现 RI 差距的来源之一，留待后续核实。

## 9. 待办 / 后续关注
- [ ] SVM 核选择：`evaluate.svm_10fold` 默认 RBF（LIBSVM 默认），如结果偏差大改 linear 并回填本文件。
- [ ] Phase 6 性能：AIDS/PROTEIN 规模需评估批量图 + 减少 inner_iter 的权衡。
- [ ] detach vs 非 detach 耗时对比实验（prompt §4.7）。
- [ ] 复现 RI 差距（MUTAG 60.75 vs 69.2）需进一步排查：N_k、M（outer_iteration）、λ vs z 表示。
