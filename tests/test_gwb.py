"""Phase 4 GW 重心 GWB 测试（SPEC §11 Phase 4 测试 1–6、prompt §9 Phase 4）。"""

import torch

from gwf.barycenter import gwb
from gwf.transport import gw_discrepancy

D = torch.float64


def _rand_sym(n: int, weighted: bool = True) -> torch.Tensor:
    A = torch.rand(n, n, dtype=D)
    A = (A + A.t()) / 2
    A = A - torch.diag(torch.diag(A))
    return A if weighted else (A > 0.5).to(D)


def _mu(n: int) -> torch.Tensor:
    return torch.full((n,), 1.0 / n, dtype=D)


# ---------------------------------------------------------------------------
# 测试 1：形状
# ---------------------------------------------------------------------------
def test_shape():
    torch.manual_seed(0)
    Nb = 15
    U_list = [_rand_sym(10), _rand_sym(8), _rand_sym(12)]
    lam = torch.tensor([0.3, 0.3, 0.4], dtype=D)
    B0 = _rand_sym(Nb)
    B, T_list = gwb(U_list, lam, B0, _mu(Nb), L=3, gwd="ppa", M=5, gamma=0.01)
    assert B.shape == (Nb, Nb)
    assert len(T_list) == 3
    for T, U in zip(T_list, U_list):
        assert T.shape == (Nb, U.shape[0])


# ---------------------------------------------------------------------------
# 测试 2：单因子恒等（K=1, λ=[1], U=C 同尺寸 → B≈U）
# ---------------------------------------------------------------------------
def test_single_factor_identity():
    torch.manual_seed(1)
    N = 12
    C = _rand_sym(N)
    m = _mu(N)
    B, T_list = gwb([C], torch.tensor([1.0], dtype=D), C, m, L=5, gwd="ppa", M=20, gamma=0.01)
    d = gw_discrepancy(B, C, m, m, 20, 0.01, "ppa")
    assert d.item() < 1e-4


# ---------------------------------------------------------------------------
# 测试 3：凸组合极限
# ---------------------------------------------------------------------------
def test_convex_combination():
    torch.manual_seed(2)
    N = 12
    U1 = _rand_sym(N)
    U2 = _rand_sym(N)
    m = _mu(N)
    B0 = _rand_sym(N)

    # λ=(1,0) → B 更接近 U1
    B_1, _ = gwb([U1, U2], torch.tensor([1.0, 0.0], dtype=D), B0, m, L=10, gwd="ppa", M=20, gamma=0.01)
    d1 = gw_discrepancy(B_1, U1, m, m, 20, 0.01, "ppa")
    d2 = gw_discrepancy(B_1, U2, m, m, 20, 0.01, "ppa")
    assert d1.item() < d2.item()

    # λ=(0.5,0.5) → d_gw(B,U1) ≈ d_gw(B,U2)
    B_h, _ = gwb([U1, U2], torch.tensor([0.5, 0.5], dtype=D), B0, m, L=10, gwd="ppa", M=20, gamma=0.01)
    dh1 = gw_discrepancy(B_h, U1, m, m, 20, 0.01, "ppa")
    dh2 = gw_discrepancy(B_h, U2, m, m, 20, 0.01, "ppa")
    assert abs(dh1.item() - dh2.item()) < 0.05


# ---------------------------------------------------------------------------
# 测试 4：L 收敛（B^(L) 变化量随 L 递减）
# ---------------------------------------------------------------------------
def test_L_convergence():
    torch.manual_seed(3)
    N = 10
    U_list = [_rand_sym(N), _rand_sym(N)]
    lam = torch.tensor([0.5, 0.5], dtype=D)
    B0 = _rand_sym(N)
    m = _mu(N)
    prev = None
    changes = []
    for L in (1, 5, 20, 50):
        B, _ = gwb(U_list, lam, B0, m, L=L, gwd="ppa", M=10, gamma=0.01)
        if prev is not None:
            changes.append((B - prev).abs().max().item())
        prev = B
    # 相邻 L 的 B 变化量应递减（收敛）
    assert all(c2 <= c1 + 1e-6 for c1, c2 in zip(changes, changes[1:]))


# ---------------------------------------------------------------------------
# 测试 5：索引方向锁定（式 21：T_k U_k^T T_k^T，T_k: B→U_k）
# ---------------------------------------------------------------------------
def test_index_direction_same_size():
    """同尺寸恒等测试：式 21 形式使 B≈U（锁定索引方向，SPEC §9 条目 1）。"""
    torch.manual_seed(4)
    N = 12
    C = _rand_sym(N)
    m = _mu(N)
    B, T_list = gwb([C], torch.tensor([1.0], dtype=D), C, m, L=5, gwd="ppa", M=20, gamma=0.01)
    # 单因子 + 恒等传输下，式 21 给出 B = U^T = U（对称邻接）
    assert torch.allclose(B, C, atol=1e-5)


def test_index_direction_diff_size():
    """不同尺寸下式 21 产生 (N_b,N_b)；转置形式在 T:(B→U_k) 约定下维度非法。"""
    torch.manual_seed(5)
    Nb = 16
    U = _rand_sym(9)
    m = _mu(Nb)
    B0 = _rand_sym(Nb)
    B, T_list = gwb([U], torch.tensor([1.0], dtype=D), B0, m, L=3, gwd="ppa", M=5, gamma=0.01)
    assert B.shape == (Nb, Nb)
    assert T_list[0].shape == (Nb, 9)
    assert B.min() >= -1e-6  # 非负（因子与传输非负）
