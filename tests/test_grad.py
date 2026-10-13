"""Phase 5 梯度正确性测试（SPEC §11 Phase 5 测试 2、§5.4、§6.3）。"""

import torch
from torch.autograd import gradcheck

from gwf.barycenter import gwb
from gwf.model import GWF

D = torch.float64


def _rand_sym(n: int) -> torch.Tensor:
    A = torch.rand(n, n, dtype=D)
    A = (A + A.t()) / 2
    return A - torch.diag(torch.diag(A))


def _fixed_T_loss(V, z, Cs, T_lists, K):
    """包络定理下的重建损失：T 固定为常数，梯度仅经 U（→V）与 λ（→z）。"""
    U = torch.sigmoid(V)
    total = torch.zeros((), dtype=V.dtype, device=V.device)
    for i, C in enumerate(Cs):
        lam = torch.softmax(z[i], dim=0)
        mu_b = torch.full((C.shape[0],), 1.0 / C.shape[0], dtype=C.dtype, device=C.device)
        outer = mu_b[:, None] @ mu_b[None, :]
        B = torch.zeros_like(C)
        for k in range(K):
            Tk = T_lists[i][k]
            B = B + lam[k] * (Tk @ U[k].t() @ Tk.t())
        B = B / outer
        total = total + torch.sum((B - C) ** 2)
    return total


def test_gradcheck():
    """∂L/∂V 与 ∂L/∂z 与数值梯度一致（T 固定，容差 1e-4，double）。"""
    torch.manual_seed(0)
    K, Nk, L, M, gamma = 2, 4, 2, 3, 0.01
    Cs = [_rand_sym(6), _rand_sym(7)]
    V = torch.randn(K, Nk, Nk, dtype=D, requires_grad=True) * 0.1
    z = torch.randn(2, K, dtype=D, requires_grad=True) * 0.1

    # 先求解并固定 T（用 detach 后的 U、λ 求解，避免建图）
    U0 = torch.sigmoid(V).detach()
    z0 = z.detach()
    T_lists = []
    for i, C in enumerate(Cs):
        lam = torch.softmax(z0[i], dim=0)
        mu_b = torch.full((C.shape[0],), 1.0 / C.shape[0], dtype=D, device=C.device)
        _, T_list = gwb([U0[k] for k in range(K)], lam, C, mu_b, L,
                        gwd="ppa", M=M, gamma=gamma)
        T_lists.append([T.detach() for T in T_list])

    fn = lambda V, z: _fixed_T_loss(V, z, Cs, T_lists, K)
    ok = gradcheck(fn, (V, z), eps=1e-6, atol=1e-4, rtol=1e-4)
    assert ok


def test_reparam_validity():
    """重参数化有效性：U=sigmoid(V)∈[0,1]，λ=softmax(z) 行和=1。"""
    torch.manual_seed(1)
    model = GWF(K=3, N_k=5, I=4, L=1, M=2).double()
    U = model.factors()
    lam = model.all_weights()
    assert (U >= 0).all() and (U <= 1).all()
    assert torch.allclose(lam.sum(dim=1), torch.ones(4, dtype=D), atol=1e-6)


def test_detach_no_graph_through_solver():
    """包络定理：T_list 不参与梯度（求解器输入已 detach），且重建仍对 V/z 有梯度。"""
    torch.manual_seed(2)
    model = GWF(K=2, N_k=4, I=1, L=2, M=3).double()
    C = _rand_sym(6)
    B, T_list, lam = model.reconstruct(C, 0)
    assert all(not T.requires_grad for T in T_list)
    # 重建 B 对 V/z 有梯度
    loss = torch.sum((B - C) ** 2)
    loss.backward()
    assert model.V.grad is not None and torch.isfinite(model.V.grad).all()
    assert model.z.grad is not None
