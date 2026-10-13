# 复现报告：Gromov-Wasserstein Factorization (GWF)，TPAMI 2022

> 论文：*Representing Graphs via Gromov-Wasserstein Factorization*（Xu, Liu, Luo, Carin）
> DOI: 10.1109/TPAMI.2022.3153126
> 本报告按 prompt.md §9 Phase 9 的交付要求组织。

## 0. 执行摘要

- 完成了 GWF 的**三条主线**：直推式 GWF、GWD 求解器（PPA + BADMM，可插拔）、归纳式 GWF-GNN。
- 全部单元测试通过（`pytest -q` 39 项全绿），覆盖 GWD 边际/等变/单调性、GWB 重心/索引方向、
  梯度 gradcheck、重参数化、数据统计校验。
- 合成数据冒烟测试：重建误差下降 53%、聚类 RI=1.000（通过 Phase 5 门槛）。
- 真实数据（MUTAG）聚类 RI 复现值 **60.75%**（论文 69.2±2.0），**偏差 8.5 个百分点，
  未落入 ±3 容差**（详见 §3 原因分析）。

## 1. 实现范围与代码结构

| 模块 | 文件 | 对应 SPEC/论文 |
| :-- | :-- | :-- |
| 数据加载 | `gwf/data.py` | SPEC §3；8 数据集统计 ±0.5% 校验通过 |
| GWD 求解器 | `gwf/transport.py` | SPEC §4.2/4.3 [Alg.1][Alg.2] |
| GW 重心 | `gwf/barycenter.py` | SPEC §4.4 [Alg.3][式 21] |
| GWF 模型 | `gwf/model.py` | SPEC §5 [式 9/12/13/16] |
| GWF-GNN | `gwf/model.py`（GINEncoder/GWFGNN） | SPEC §5.3 [式 15] |
| FGW | `gwf/model.py`（FGW） | SPEC §1.2 [式 25/26] |
| 训练 | `gwf/train.py` | SPEC §7 [Alg.4] |
| 评估 | `gwf/evaluate.py` | SPEC §8 |
| 测试 | `tests/test_*.py` | SPEC §10 溯源表 |

## 2. 关键实现决策（[待确认] 项结论）

| prompt §11 条目 | 结论 | 依据 |
| :-- | :-- | :-- |
| 1 索引方向 | 式 21 `T_k U_k^T T_k^T`，`T_k: B→U_k` | 同尺寸恒等测试锁定 |
| 2 N_k | 默认 10（=min_i N_i for MUTAG） | 敏感度实验未完成 |
| 3 per-graph vs batch | per-graph（最忠实 Alg.4） | — |
| 4 alternating | 先 z 后 V，各 1 步 | 论文结论 |
| 5 IMDB GNN 输入 | 全 1 向量 | 默认，消融未完成 |
| 6 z 初始化 | N(0, 0.01) | — |
| 7 损失归一化 | 原始 ‖·‖_F²（不归一化） | — |
| 8 Sinkhorn 平移 | log-domain logsumexp，平移吸收全局常数 | NOTES §3.2 |
| 9 K-means 表示 | λ=softmax(z) | 与 z 对照见 §3 |
| 10 γ=0.01 数值风险 | log-domain Sinkhorn，无 NaN | NOTES §3.2 |

## 3. 复现结果对照

### 3.1 图聚类（Rand-Index %，论文表 7.3）

| 数据集 | 复现 GWF_PPA | 论文 GWF_PPA | 偏差 |
| :-- | :-- | :-- | :-- |
| MUTAG (K=15) | 60.75 | 69.2±2.0 | **-8.45** |

> 其余数据集、多 seed、BADMM、GWF-GNN 列因计算资源限制未全部完成（见 §6）。

### 3.2 偏差原因分析（MUTAG）

复现值 60.75 低于论文 69.2，偏差 8.45 个百分点。已排查并排除以下因素：

1. **实现正确性**：gradcheck 通过（∂L/∂V、∂L/∂z 与数值梯度一致）、合成数据 RI=1.0、
   GWD 边际/等变/单调性测试全绿。核心数学（C_XY、PPA/BADMM、GWB、包络定理 detach）正确。
2. **数据**：8 数据集统计与论文表 7.1 逐列 ±0.5% 一致。

仍可能造成偏差、尚未完全锁定的因素：

- **图因子尺寸 N_k**：本实现用 10（prompt 默认）。论文仅说"预先设定"未给数值，若其实际
  N_k 更大（接近图节点数），重建更精确、聚类更高。
- **GWD 迭代配置**：论文 M=20；官方仓库（`cmp_algorithm_gwl.py`）在算法对比中使用
  `outer_iteration=3000` 至收敛，与"训练用 M=20"的映射未完全确认。
- **K-means 表示**：本实现用 λ=softmax(z)（prompt 默认），论文 §7.3 对 λ vs z 未明确。
  该对照实验见 §3.3。
- **预热初始化细节**：N_k 与采样图尺寸不匹配时的处理（本实现取左上角子块）为自主决策。

### 3.3 λ vs z 表示对照（K-means）

（待 λ/z 实验完成后填写）

## 4. 已通过的验证（Phase 1–5）

| Phase | 结果 |
| :-- | :-- |
| 1 | SPEC v1.0 冻结 |
| 2 | `tests/test_data.py` 11 项全绿；8 数据集统计 ±0.5% |
| 3 | `tests/test_gwd.py` 全绿（边际/无梯度/恒等/置换等变/单调/退化） |
| 4 | `tests/test_gwb.py` 6 项全绿（含索引方向锁定） |
| 5 | `tests/test_grad.py` 全绿（gradcheck 1e-4、detach、重参数化）；合成冒烟 RI=1.0 |

## 5. 数值稳定关键决策（详见 NOTES.md）

- PPA（γ=0.01）内层 Sinkhorn 用 **log-domain logsumexp**（1/γ=100 使 Gibbs 核动态范围
  达 ~1e-174，线性域 float32 会下溢、精度差）。
- BADMM（γ=1）动态范围温和，线性域 + 最后 Sinkhorn 平衡。
- 同尺寸因子批量 GWD（向量化 K 个传输），提速 ~5×。

## 6. 未完成项与原因

- **Phase 6 完整聚类表**（7 数据集 × PPA/BADMM × K∈{15,30} × 5 seeds）：单次 MUTAG 交替
  训练约 66 分钟（CPU），完整表格约需数百小时，超出本次资源。
- **Phase 7 GWF-GNN 验证**：代码已实现，端到端数值验证未完成。
- **Phase 8 扩展**：FGW/半监督/敏感性/可视化未完成。
- **交替 vs 联合对比图**：未产出（需两次完整训练）。

## 7. SPEC 变更历史

| 版本 | 变更 | 影响 |
| :-- | :-- | :-- |
| v1.0 | 冻结 | — |

（实现过程中对 SPEC 无静默变更；`inner_iter` 为新增实现细节，已记 NOTES。）
