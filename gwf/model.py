"""GWF 直推式模型（式 9/12/13/16/27）与 FGW 扩展。

对应 SPEC §5（模型规格）、§6（目标函数与梯度）。

可学习参数（logit 空间，SPEC §5.1）：
- ``V``: ``(K, N_k, N_k)`` 因子 logit，``U = sigmoid(V)`` [式 12]。
- ``z``: ``(I, K)`` 图权重 logit，``λ_i = softmax(z_i)`` [式 13]。
- ``H``（FGW 版）: ``(K, N_k, D)`` 因子节点嵌入 [式 26]。

包络定理（SPEC §5.4）：前向求解最优传输时**对因子与权重 detach**（不沿 Sinkhorn 迭代
反传），重建项中 T_k 视为常数（``T_k.detach()``），梯度只经 U（→V）与 λ（→z）流动。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .barycenter import gwb
from .transport import gwd_ppa, gwd_badmm


class GWF(nn.Module):
    """直推式 GWF：可学习图因子 V + 每图权重 z。"""

    def __init__(self, K: int, N_k: int, I: int, L: int = 5, M: int = 20,
                 gamma: float = 0.01, gwd: str = "ppa", inner_iter: int = 50,
                 loss_normalize: bool = False, seed: int | None = None):
        super().__init__()
        self.K, self.N_k, self.I = K, N_k, I
        self.L, self.M, self.gamma, self.gwd = L, M, gamma, gwd
        self.inner_iter = inner_iter
        self.loss_normalize = loss_normalize
        self._gwd_fn = gwd_ppa if gwd == "ppa" else gwd_badmm
        if seed is not None:
            torch.manual_seed(seed)
        # logit 空间可学习参数
        self.V = nn.Parameter(torch.randn(K, N_k, N_k) * 0.1)
        self.z = nn.Parameter(torch.randn(I, K) * 0.1)  # N(0, 0.01) → std 0.1

    # -- 参数映射 -----------------------------------------------------------
    def factors(self) -> torch.Tensor:
        """U = sigmoid(V)，shape (K, N_k, N_k)。[式 12]"""
        return torch.sigmoid(self.V)

    def weights(self, i: int) -> torch.Tensor:
        """λ_i = softmax(z_i)，shape (K,)。[式 13]"""
        return torch.softmax(self.z[i], dim=0)

    def all_weights(self) -> torch.Tensor:
        """全部 λ，shape (I, K)。"""
        return torch.softmax(self.z, dim=1)

    # -- 预热初始化 [Alg.4] ------------------------------------------------
    def warm_init_V(self, C_list: list[torch.Tensor], indices: list[int]) -> None:
        """V_k = logit(0.98*C_k + 0.01)，C_k 为随机抽取的 K 个图（SPEC §7.2）。

        因子尺寸 N_k 与采样图尺寸可能不同：取采样图左上角 ``(N_k, N_k)`` 子块
        （若采样图更小则用其自身，缺失处填 0）[决策，见 NOTES.md §11 条目 2]。
        """
        device = self.V.device
        Nk = self.N_k
        with torch.no_grad():
            for k, idx in enumerate(indices):
                C = C_list[idx].to(device)
                n = C.shape[0]
                sub = torch.zeros(Nk, Nk, dtype=C.dtype, device=device)
                m = min(n, Nk)
                sub[:m, :m] = C[:m, :m]
                x = torch.clamp(0.98 * sub + 0.01, 1e-6, 1 - 1e-6)
                self.V[k] = torch.log(x / (1 - x))

    # -- 前向（单图 i）------------------------------------------------------
    def reconstruct(self, C_i: torch.Tensor, i: int) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """计算 GW 重心 B^L 与最优传输 T_{1:K}^L。

        Args:
            C_i: 观测图邻接矩阵 (N_i, N_i)。
            i: 图索引（用于取 z_i）。
        Returns:
            ``(B, T_list, lam_i)``。
        """
        U = self.factors()                       # (K,N_k,N_k) 含梯度
        lam = self.weights(i)                     # (K,) 含梯度
        U_det = U.detach()
        lam_det = lam.detach()

        N_i = C_i.shape[0]
        mu_b = torch.full((N_i,), 1.0 / N_i, dtype=C_i.dtype, device=C_i.device)
        U_list = [U_det[k] for k in range(self.K)]
        B, T_list = gwb(U_list, lam_det, C_i, mu_b, self.L,
                        gwd=self.gwd, M=self.M, gamma=self.gamma, inner_iter=self.inner_iter)

        # 重建项（包络定理）：T 视为常数，梯度经 U、λ 流动
        outer = mu_b[:, None] @ mu_b[None, :]
        B = torch.zeros_like(C_i)
        for k in range(self.K):
            Tk = T_list[k].detach()
            B = B + lam[k] * (Tk @ U[k].t() @ Tk.t())
        B = B / outer
        return B, T_list, lam

    def loss(self, C_i: torch.Tensor, i: int) -> torch.Tensor:
        """L_i = ||B^L - C_i||_F^2 [式 27]，可选 /N_i^2 归一化。"""
        B, _, _ = self.reconstruct(C_i, i)
        loss = torch.sum((B - C_i) ** 2)
        if self.loss_normalize:
            loss = loss / (C_i.shape[0] ** 2)
        return loss

    def embed(self, i: int | None = None) -> torch.Tensor:
        """返回表示 λ（softmax 后）。i=None 返回全部 (I,K)，否则 (K,)。"""
        w = self.all_weights()
        return w if i is None else w[i]


class FGW(GWF):
    """带节点属性的 FGW 版本（式 25/26，SPEC §1.2 P1）。"""

    def __init__(self, K: int, N_k: int, I: int, D: int, alpha: float = 1.0,
                 L: int = 5, M: int = 20, gamma: float = 0.01, gwd: str = "ppa",
                 loss_normalize: bool = False, seed: int | None = None):
        super().__init__(K, N_k, I, L, M, gamma, gwd, loss_normalize, seed)
        self.D = D
        self.alpha = alpha
        self.H = nn.Parameter(torch.randn(K, N_k, D) * 0.1)  # 因子节点嵌入

    def reconstruct(self, C_i, F_i, i):
        """FGW 前向：返回 (B, F_b, T_list, lam)。"""
        U = self.factors()
        H = self.H
        lam = self.weights(i)
        U_det, H_det, lam_det = U.detach(), H.detach(), lam.detach()

        N_i = C_i.shape[0]
        mu_b = torch.full((N_i,), 1.0 / N_i, dtype=C_i.dtype, device=C_i.device)
        # FGW：C_XY 替换为 C_XY + D（D 由特征构造，式 25 后文）
        # 此处调用 gwb 需传入 cost_extra；为简化，先实现结构重建 + 特征更新
        B, T_list = gwb(U_det, lam_det, C_i, mu_b, self.L,
                        gwd=self.gwd, M=self.M, gamma=self.gamma, inner_iter=self.inner_iter)
        outer = mu_b[:, None] @ mu_b[None, :]
        B = torch.zeros_like(C_i)
        F_b = torch.zeros(N_i, self.D, dtype=C_i.dtype, device=C_i.device)
        for k in range(self.K):
            Tk = T_list[k].detach()
            B = B + lam[k] * (Tk @ U[k].t() @ Tk.t())
            F_b = F_b + lam[k] * (Tk @ H[k])
        B = B / outer
        F_b = F_b / mu_b[:, None]  # 式 26：除以 μ_b 1_D^T
        return B, F_b, T_list, lam

    def loss(self, C_i, F_i, i):
        B, F_b, _, _ = self.reconstruct(C_i, F_i, i)
        loss = torch.sum((B - C_i) ** 2) + self.alpha * torch.sum((F_b - F_i) ** 2)
        if self.loss_normalize:
            loss = loss / (C_i.shape[0] ** 2)
        return loss


class _GINConv(nn.Module):
    """GIN 卷积层：h' = MLP((1+ε)·h + A@h)（GIN, Xu et al. 2019）。"""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = (1.0 + self.eps) * h + A @ h
        return self.mlp(x)


class GINEncoder(nn.Module):
    """图同构网络编码器：n 层 GIN + 全局均值池化 + 线性输出 K 维。

    输出经 softmax 后作为 GWF 的权重 λ [式 15]。无节点属性时用全 1 或节点度作为输入
    （SPEC §9 条目 5）。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 5,
                 eps: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            self.convs.append(_GINConv(d_in, hidden_dim, eps))
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        h = F
        for conv in self.convs:
            h = conv(A, h)
        h = h.mean(dim=0)  # 全局均值池化
        return self.readout(h)  # (K,)


class GWFGNN(nn.Module):
    """归纳式 GWF-GNN [式 15]：GNN 编码器（输出 λ）+ GWF 解码器（因子 + 重心）。"""

    def __init__(self, K: int, N_k: int, encoder: nn.Module, L: int = 5, M: int = 20,
                 gamma: float = 0.01, gwd: str = "ppa", inner_iter: int = 50,
                 loss_normalize: bool = False):
        super().__init__()
        self.K, self.N_k = K, N_k
        self.L, self.M, self.gamma, self.gwd = L, M, gamma, gwd
        self.inner_iter = inner_iter
        self.loss_normalize = loss_normalize
        self.encoder = encoder
        self.V = nn.Parameter(torch.randn(K, N_k, N_k) * 0.1)

    def factors(self) -> torch.Tensor:
        return torch.sigmoid(self.V)

    def warm_init_V(self, C_list: list[torch.Tensor], indices: list[int]) -> None:
        device = self.V.device
        Nk = self.N_k
        with torch.no_grad():
            for k, idx in enumerate(indices):
                C = C_list[idx].to(device)
                n = C.shape[0]
                sub = torch.zeros(Nk, Nk, dtype=C.dtype, device=device)
                m = min(n, Nk)
                sub[:m, :m] = C[:m, :m]
                x = torch.clamp(0.98 * sub + 0.01, 1e-6, 1 - 1e-6)
                self.V[k] = torch.log(x / (1 - x))

    def weights_from_graph(self, C_i: torch.Tensor, F_i: torch.Tensor | None) -> torch.Tensor:
        if F_i is None:
            F_i = torch.ones(C_i.shape[0], 1, dtype=C_i.dtype, device=C_i.device)
        return torch.softmax(self.encoder(C_i, F_i), dim=0)

    def reconstruct(self, C_i: torch.Tensor, F_i: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        U = self.factors()
        lam = self.weights_from_graph(C_i, F_i)
        U_det = U.detach()
        lam_det = lam.detach()

        N_i = C_i.shape[0]
        mu_b = torch.full((N_i,), 1.0 / N_i, dtype=C_i.dtype, device=C_i.device)
        B, T_list = gwb([U_det[k] for k in range(self.K)], lam_det, C_i, mu_b, self.L,
                        gwd=self.gwd, M=self.M, gamma=self.gamma, inner_iter=self.inner_iter)

        outer = mu_b[:, None] @ mu_b[None, :]
        B = torch.zeros_like(C_i)
        for k in range(self.K):
            Tk = T_list[k].detach()
            B = B + lam[k] * (Tk @ U[k].t() @ Tk.t())
        B = B / outer
        return B, T_list, lam

    def loss(self, C_i: torch.Tensor, F_i: torch.Tensor | None = None) -> torch.Tensor:
        B, _, _ = self.reconstruct(C_i, F_i)
        loss = torch.sum((B - C_i) ** 2)
        if self.loss_normalize:
            loss = loss / (C_i.shape[0] ** 2)
        return loss
