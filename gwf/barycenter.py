"""GW 重心 GWB [Alg.3][式 21]。

对应 SPEC §4.4。给定 K 个图因子 U_{1:K}、权重 λ、初始重心 B^0 与节点分布 μ_b，
迭代 L 步：每步对每个因子求解最优传输 T_k = GWD(B, U_k)（X 侧 = B，Y 侧 = U_k），
再按闭式更新 B = (1/μ_b μ_b^T) Σ_k λ_k T_k U_k^T T_k^T [式 21]。

索引方向（SPEC §9 条目 1）：T_k 形状 (N_b, N_k)，映射 B→U_k；闭式更新用
``T_k U_k^T T_k^T``（论文式 21）。``.pdf/abtract.md`` 中的 ``T_k^T U_k T_k`` 是
T_k: U_k→B 的转置约定，与本实现的 GWD 输出约定（T: X→Y）一致时二者等价；
本实现采用式 21，由 Phase 4 同尺寸恒等测试锁定。

同尺寸因子时使用批量 GWD（向量化 K 个传输），输出与循环版误差 < 1e-8（SPEC §4.4）。
"""

from __future__ import annotations

from typing import Sequence

import torch

from .transport import gwd_badmm, gwd_badmm_batched, gwd_ppa, gwd_ppa_batched


def gwb(U_list: Sequence[torch.Tensor], lam: torch.Tensor, B0: torch.Tensor,
        mu_b: torch.Tensor, L: int, gwd: str = "ppa", **kw) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """GWB [Alg.3]：迭代 L 步求 GW 重心。

    Args:
        U_list: K 个图因子，每个 ``(N_k, N_k)``（尺寸可不同）。
        lam: 权重 ``(K,)``，Σ_k lam_k = 1（softmax 后）。
        B0: 初始重心 ``(N_b, N_b)``。
        mu_b: 重心节点分布 ``(N_b,)``（均匀）。
        L: 迭代步数（默认 5）。
        gwd: 'ppa' | 'badmm'。
        **kw: 透传给 GWD 的超参（M, gamma, cost_extra, inner_iter）。
    Returns:
        ``(B, T_list)``：B 为 ``(N_b, N_b)``，T_list 为 K 个 ``(N_b, N_k)``。
    """
    K = len(U_list)
    B = B0
    same_size = all(U.shape == U_list[0].shape for U in U_list)
    outer = mu_b[:, None] @ mu_b[None, :]

    for _ in range(L):
        if same_size:
            U_stack = torch.stack([U.detach() for U in U_list])  # (K, N_k, N_k)
            Nk = U_stack.shape[1]
            mu_k = torch.full((Nk,), 1.0 / Nk, dtype=U_stack.dtype, device=U_stack.device)
            fn = gwd_ppa_batched if gwd == "ppa" else gwd_badmm_batched
            T_stack = fn(B, U_stack, mu_b, mu_k, **kw)          # (K, N_b, N_k)
            # 闭式更新 [式 21]：T_k U_k^T T_k^T
            TU = T_stack @ U_stack.transpose(-1, -2)            # (K, N_b, N_k)
            B_stack = TU @ T_stack.transpose(-1, -2)            # (K, N_b, N_b)
            B = (lam[:, None, None] * B_stack).sum(dim=0) / outer
            T_list = [T_stack[k] for k in range(K)]
        else:
            fn = gwd_ppa if gwd == "ppa" else gwd_badmm
            T_list = []
            acc = torch.zeros_like(B)
            for U, lamk in zip(U_list, lam):
                mu_k = torch.full((U.shape[0],), 1.0 / U.shape[0], dtype=U.dtype, device=U.device)
                T = fn(B, U, mu_b, mu_k, **kw)
                T_list.append(T)
                acc = acc + lamk * (T @ U.t() @ T.t())
            B = acc / outer
    return B, T_list
