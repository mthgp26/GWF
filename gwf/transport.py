"""GWD 求解器：PPA [Alg.1] 与 BADMM [Alg.2]，以及 GW 差异与熵正则基线。

对应 SPEC §4。所有 Sinkhorn 类迭代都在 log-space 或带归一化下实现，
禁止裸 ``exp``（SPEC §4.2/4.3 数值稳定策略、prompt §13.6、§11 条目 10）。

关键约定（SPEC §2.2）：
- 传输多面体 ``T``：``T >= 0``、``T @ 1 = mu_X``（行边际，X 侧）、``T.T @ 1 = mu_Y``（列边际，Y 侧）。
- X 侧 = 观测图/重心 B，Y 侧 = 图因子 U_k。

数值稳定策略：
- PPA（γ=0.01，1/γ=100）Gibbs 核动态范围可达 ~1e-174，故**内层 Sinkhorn 全程在 log 域**
  用 ``logsumexp`` 计算（log-domain Sinkhorn-Knopp），避免 exp 上溢/下溢。
- BADMM（γ=1）动态范围温和，线性域计算即可；最后做一次线性 Sinkhorn 平衡以精确满足双边际。

实现说明（与官方仓库 SDS-Lab/Gromov-Wasserstein-Factorization 交叉核对）：
- PPA 的"一步 Sinkhorn-Knopp"在官方实现中为**固定 kernel + 内层 Sinkhorn 迭代至收敛**
  （``sinkhorn_knopp_iteration`` 内层 50 步），否则列边际无法满足（见 NOTES.md）。
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def _make_cxy(C_X: torch.Tensor, C_Y: torch.Tensor, mu_X: torch.Tensor,
              mu_Y: torch.Tensor) -> torch.Tensor:
    """构造 GW 二次型线性项 C_XY [式 7]。

    C_XY = (C_X⊙C_X) μ_X 1_NY^T + 1_NX μ_Y^T (C_Y⊙C_Y)^T，shape (N_X, N_Y)。
    """
    Nx = C_X.shape[0]
    Ny = C_Y.shape[0]
    ones_y = torch.ones(1, Ny, dtype=C_X.dtype, device=C_X.device)
    term1 = (C_X * C_X) @ mu_X[:, None] @ ones_y          # (Nx,1)@(1,Ny)
    term2 = torch.ones(Nx, 1, dtype=C_X.dtype, device=C_X.device) @ mu_Y[None, :] @ (C_Y * C_Y).t()
    return term1 + term2


def _sinkhorn_balance(K: torch.Tensor, mu_X: torch.Tensor, mu_Y: torch.Tensor,
                      max_iter: int = 500, tol: float = 1e-9) -> torch.Tensor:
    """线性域交替行列缩放，使非负矩阵 K 满足双边际。返回 ``diag(a) K diag(b)``。

    仅用于动态范围温和的矩阵（如 BADMM γ=1 的 T）。
    """
    a = torch.ones_like(mu_X)
    b = torch.ones_like(mu_Y)
    for _ in range(max_iter):
        a_new = mu_X / (K @ b + _EPS)
        b_new = mu_Y / (K.t() @ a_new + _EPS)
        err = (a_new - a).abs().max() + (b_new - b).abs().max()
        a, b = a_new, b_new
        if err < tol:
            break
    return a[:, None] * K * b[None, :]


def _log_sinkhorn(log_K: torch.Tensor, mu_X: torch.Tensor, mu_Y: torch.Tensor,
                  max_iter: int = 500, tol: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """log-domain Sinkhorn-Knopp：平衡 ``exp(log_K)`` 到双边际。

    返回 ``(log_a, log_b)``，使 ``T = exp(log_a[:,None] + log_K + log_b[None,:])``
    满足 ``T@1=mu_X``、``T.T@1=mu_Y``。全程 log 域，稳健处理 γ=0.01 的极端动态范围。
    """
    log_muX = mu_X.log()
    log_muY = mu_Y.log()
    log_a = torch.zeros_like(mu_X)
    log_b = torch.zeros_like(mu_Y)
    for _ in range(max_iter):
        log_a = log_muX - torch.logsumexp(log_K + log_b[None, :], dim=1)
        log_b = log_muY - torch.logsumexp(log_K + log_a[:, None], dim=0)
        row = torch.exp(torch.logsumexp(log_K + log_b[None, :], dim=1) + log_a)
        col = torch.exp(torch.logsumexp(log_K + log_a[:, None], dim=0) + log_b)
        if ((row - mu_X).abs().max() + (col - mu_Y).abs().max()).item() < tol:
            break
    return log_a, log_b


def gwd_ppa(C_X: torch.Tensor, C_Y: torch.Tensor, mu_X: torch.Tensor,
            mu_Y: torch.Tensor, M: int, gamma: float,
            cost_extra: torch.Tensor | None = None,
            inner_iter: int = 500) -> torch.Tensor:
    """GWD_PPA [Alg.1]：近端点算法 + log-domain Sinkhorn-Knopp 内层迭代。

    Args:
        C_X, C_Y: 邻接矩阵 (N_X,N_X) / (N_Y,N_Y)。
        mu_X, mu_Y: 节点分布 (N_X,) / (N_Y,)。
        M: 外层迭代次数。gamma: 正则化权重（PPA=0.01）。
        cost_extra: 可选附加代价 (N_X,N_Y)（FGW 用，SPEC §4.10）。
        inner_iter: 内层 Sinkhorn 最大迭代次数。
    Returns:
        T (N_X, N_Y)，双边际满足（近似）。
    """
    C_XY = _make_cxy(C_X, C_Y, mu_X, mu_Y)
    if cost_extra is not None:
        C_XY = C_XY + cost_extra

    T = mu_X[:, None] @ mu_Y[None, :]     # T^(0)，全正
    for _ in range(M):
        cost = C_XY - 2.0 * (C_X @ T @ C_Y.t())
        # log 域 kernel：log K = -cost/gamma + log T^(m)
        log_K = -cost / gamma + torch.log(T)
        log_a, log_b = _log_sinkhorn(log_K, mu_X, mu_Y, max_iter=inner_iter)
        T = torch.exp(log_a[:, None] + log_K + log_b[None, :])
    return T


def _log_sinkhorn_batched(log_K: torch.Tensor, mu_X: torch.Tensor, mu_Y: torch.Tensor,
                          max_iter: int = 500, tol: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """批量 log-domain Sinkhorn：``log_K`` shape ``(K, Nx, Ny)``，双边际 ``mu_X``/``mu_Y``。

    返回 ``(log_a, log_b)`` shape ``(K, Nx)`` / ``(K, Ny)``。
    """
    K, Nx, Ny = log_K.shape
    log_muX = mu_X.log().view(1, Nx)
    log_muY = mu_Y.log().view(1, Ny)
    log_a = torch.zeros(K, Nx, dtype=log_K.dtype, device=log_K.device)
    log_b = torch.zeros(K, Ny, dtype=log_K.dtype, device=log_K.device)
    for _ in range(max_iter):
        log_a = log_muX - torch.logsumexp(log_K + log_b[:, None, :], dim=2)
        log_b = log_muY - torch.logsumexp(log_K + log_a[:, :, None], dim=1)
        row = torch.exp(torch.logsumexp(log_K + log_b[:, None, :], dim=2) + log_a)
        col = torch.exp(torch.logsumexp(log_K + log_a[:, :, None], dim=1) + log_b)
        if ((row - mu_X).abs().max() + (col - mu_Y).abs().max()).item() < tol:
            break
    return log_a, log_b


def gwd_ppa_batched(C_X: torch.Tensor, C_Y: torch.Tensor, mu_X: torch.Tensor,
                    mu_Y: torch.Tensor, M: int, gamma: float,
                    cost_extra: torch.Tensor | None = None,
                    inner_iter: int = 50) -> torch.Tensor:
    """批量 GWD_PPA：``C_Y`` shape ``(K, Ny, Ny)``（同尺寸因子），返回 ``T`` ``(K, Nx, Ny)``。

    与 ``gwd_ppa`` 同算法 [Alg.1]，但对 K 个因子批量计算（SPEC §4.4 向量化优化）。
    """
    K, Ny, _ = C_Y.shape
    Nx = C_X.shape[0]
    dev, dt = C_X.device, C_X.dtype

    ones_y = torch.ones(1, Ny, dtype=dt, device=dev)
    term1 = ((C_X * C_X) @ mu_X[:, None] @ ones_y).unsqueeze(0).expand(K, -1, -1)
    ones_x = torch.ones(Nx, 1, dtype=dt, device=dev)
    term2 = ones_x @ mu_Y[None, :] @ (C_Y * C_Y).transpose(-1, -2)
    C_XY = term1 + term2
    if cost_extra is not None:
        C_XY = C_XY + cost_extra

    T = (mu_X[:, None] @ mu_Y[None, :]).unsqueeze(0).expand(K, -1, -1)
    for _ in range(M):
        cost = C_XY - 2.0 * (C_X @ T) @ C_Y.transpose(-1, -2)
        log_K = -cost / gamma + torch.log(T)
        log_a, log_b = _log_sinkhorn_batched(log_K, mu_X, mu_Y, max_iter=inner_iter)
        T = torch.exp(log_a[:, :, None] + log_K + log_b[:, None, :])
    return T


def gwd_badmm(C_X: torch.Tensor, C_Y: torch.Tensor, mu_X: torch.Tensor,
              mu_Y: torch.Tensor, M: int, gamma: float,
              cost_extra: torch.Tensor | None = None) -> torch.Tensor:
    """GWD_BADMM [Alg.2]：Bregman ADMM。

    Args 同 gwd_ppa。gamma：BADMM=1.0。
    """
    C_XY = _make_cxy(C_X, C_Y, mu_X, mu_Y)
    if cost_extra is not None:
        C_XY = C_XY + cost_extra

    Nx, Ny = C_X.shape[0], C_Y.shape[0]
    ones_x = torch.ones(Nx, dtype=C_X.dtype, device=C_X.device)
    ones_y = torch.ones(Ny, dtype=C_X.dtype, device=C_X.device)

    T = mu_X[:, None] @ mu_Y[None, :]
    Z = torch.zeros(Nx, Ny, dtype=C_X.dtype, device=C_X.device)
    for _ in range(M):
        E1 = (2.0 * (C_X.t() @ T @ C_Y) + Z) / gamma
        E1 = E1 - E1.max()                 # exp(+) 必做最大值平移
        Phi1 = torch.exp(E1) * T
        S = Phi1 * (mu_Y / (Phi1.t() @ ones_x + _EPS))[None, :]   # 列归一化

        E2 = -(C_XY - 2.0 * (C_X @ S @ C_Y.t()) + Z) / gamma
        E2 = E2 - E2.max()
        Phi2 = torch.exp(E2) * S
        T = (mu_X / (Phi2 @ ones_y + _EPS))[:, None] * Phi2       # 行归一化

        Z = Z + gamma * (T - S)

    # 最后 Sinkhorn 平衡：T 行边际已精确、列边际近似，平衡后双边际精确
    T = _sinkhorn_balance(T, mu_X, mu_Y)
    return T


def gwd_badmm_batched(C_X: torch.Tensor, C_Y: torch.Tensor, mu_X: torch.Tensor,
                      mu_Y: torch.Tensor, M: int, gamma: float,
                      cost_extra: torch.Tensor | None = None) -> torch.Tensor:
    """批量 GWD_BADMM：``C_Y`` shape ``(K, Ny, Ny)``，返回 ``T`` ``(K, Nx, Ny)``。"""
    K, Ny, _ = C_Y.shape
    Nx = C_X.shape[0]
    dev, dt = C_X.device, C_X.dtype

    ones_y = torch.ones(1, Ny, dtype=dt, device=dev)
    term1 = ((C_X * C_X) @ mu_X[:, None] @ ones_y).unsqueeze(0).expand(K, -1, -1)
    ones_x = torch.ones(Nx, 1, dtype=dt, device=dev)
    term2 = ones_x @ mu_Y[None, :] @ (C_Y * C_Y).transpose(-1, -2)
    C_XY = term1 + term2
    if cost_extra is not None:
        C_XY = C_XY + cost_extra

    ones_nx = torch.ones(Nx, dtype=dt, device=dev)
    ones_ny = torch.ones(Ny, dtype=dt, device=dev)

    T = (mu_X[:, None] @ mu_Y[None, :]).unsqueeze(0).expand(K, -1, -1)
    Z = torch.zeros(K, Nx, Ny, dtype=dt, device=dev)
    for _ in range(M):
        E1 = (2.0 * (C_X.t() @ T @ C_Y) + Z) / gamma
        E1 = E1 - E1.amax(dim=(1, 2), keepdim=True)
        Phi1 = torch.exp(E1) * T
        S = Phi1 * (mu_Y / (Phi1.transpose(-1, -2) @ ones_nx + _EPS))[:, None, :]

        E2 = -(C_XY - 2.0 * (C_X @ S @ C_Y.transpose(-1, -2)) + Z) / gamma
        E2 = E2 - E2.amax(dim=(1, 2), keepdim=True)
        Phi2 = torch.exp(E2) * S
        T = (mu_X / (Phi2 @ ones_ny + _EPS))[:, :, None] * Phi2

        Z = Z + gamma * (T - S)

    # 最后批量 Sinkhorn 平衡
    log_T = torch.log(T.clamp(min=1e-300))
    la, lb = _log_sinkhorn_batched(log_T, mu_X, mu_Y, max_iter=500)
    return torch.exp(la[:, :, None] + log_T + lb[:, None, :])


def gw_discrepancy(C_X: torch.Tensor, C_Y: torch.Tensor, mu_X: torch.Tensor,
                   mu_Y: torch.Tensor, M: int, gamma: float,
                   gwd: str = "ppa",
                   cost_extra: torch.Tensor | None = None) -> torch.Tensor:
    """计算 GW 差异 d_gw = <C_XY - 2 C_X T C_Y^T, T>^{1/2} [式 7]。

    Args:
        gwd: 'ppa' | 'badmm'。
    Returns:
        标量 tensor（已 detach，不建图）。负值被 clamp 到 0 再开方。
    """
    fn = gwd_ppa if gwd == "ppa" else gwd_badmm
    T = fn(C_X, C_Y, mu_X, mu_Y, M, gamma, cost_extra)
    C_XY = _make_cxy(C_X, C_Y, mu_X, mu_Y)
    if cost_extra is not None:
        C_XY = C_XY + cost_extra
    val = torch.sum((C_XY - 2.0 * (C_X @ T @ C_Y.t())) * T)
    val = torch.clamp(val, min=0.0)
    return val.sqrt().detach()


def entropic_gw_baseline(C_X: torch.Tensor, C_Y: torch.Tensor, mu_X: torch.Tensor,
                         mu_Y: torch.Tensor, eps: float = 1.0,
                         max_iter: int = 100, tol: float = 1e-9) -> torch.Tensor:
    """熵正则化 GW 基线（Peyré et al. 2016 / Solomon et al. 2016），用于 Phase 3 对照。

    迭代 ``T = Sinkhorn(exp(-(C_XY - 2 C_X T C_Y^T)/eps), mu_X, mu_Y)`` 至收敛。
    仅用于定性对照，不作为主线求解器。
    """
    C_XY = _make_cxy(C_X, C_Y, mu_X, mu_Y)
    T = mu_X[:, None] @ mu_Y[None, :]
    for _ in range(max_iter):
        cost = C_XY - 2.0 * (C_X @ T @ C_Y.t())
        E = -cost / eps
        E = E - E.max()
        K = torch.exp(E)
        T_new = _sinkhorn_balance(K, mu_X, mu_Y)
        if torch.norm(T_new - T) < tol:
            T = T_new
            break
        T = T_new
    return T
