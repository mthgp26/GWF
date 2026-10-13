"""GWF 训练（Algorithm 4，joint / alternating），含包络定理 detach。对应 SPEC §7。"""

from __future__ import annotations

import torch

from .model import GWF


def train_gwf(model: GWF, graphs, lr: float = 0.01, epochs: int = 10,
              optim: str = "alternating", alt_inner: int = 1, device: str = "cpu",
              verbose: bool = False) -> list[float]:
    """Algorithm 4 训练循环（per-graph SGD + Adam）。

    Args:
        model: GWF 模型（已 warm_init）。
        graphs: ``list[Graph]`` 或 ``list[Tensor]``（邻接矩阵）。
        lr, epochs: Adam 学习率与轮数（论文 0.01 / 10）。
        optim: 'joint' | 'alternating'。
        alt_inner: 交替优化每轮 inner step 数（默认 1，SPEC §7.4）。
        device: 'mps' | 'cpu'。
    Returns:
        ``losses``：每 epoch 的平均损失列表。
    """
    model.to(device)
    Cs = [g.C.to(device) if hasattr(g, "C") else g.to(device) for g in graphs]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, C_i in enumerate(Cs):
            if optim == "joint":
                optimizer.zero_grad()
                loss = model.loss(C_i, i)
                loss.backward()
                optimizer.step()
            else:  # alternating
                for _ in range(alt_inner):
                    # 1) 冻结 V，更新 z_i
                    optimizer.zero_grad()
                    loss = model.loss(C_i, i)
                    loss.backward()
                    model.V.grad = None
                    optimizer.step()
                for _ in range(alt_inner):
                    # 2) 冻结 z，更新 V
                    optimizer.zero_grad()
                    loss = model.loss(C_i, i)
                    loss.backward()
                    model.z.grad = None
                    optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(Cs)
        losses.append(avg)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} loss={avg:.6f}")
    return losses


def train_gwf_gnn(model, graphs, lr: float = 0.01, epochs: int = 10,
                  device: str = "cpu", verbose: bool = False) -> list[float]:
    """GWF-GNN 训练（联合优化 V 与 GNN θ）。[式 15 + Alg.4 修改版]"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for g in graphs:
            C = g.C.to(device)
            F = g.F.to(device) if g.F is not None else None
            optimizer.zero_grad()
            loss = model.loss(C, F)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(graphs)
        losses.append(avg)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} loss={avg:.6f}")
    return losses


def embed_gnn(model, graphs, device: str = "cpu") -> torch.Tensor:
    """GWF-GNN 的图表示：对每图前向得 λ，返回 (I, K)。"""
    model.to(device)
    model.eval()
    embs = []
    with torch.no_grad():
        for g in graphs:
            C = g.C.to(device)
            F = g.F.to(device) if g.F is not None else None
            embs.append(model.weights_from_graph(C, F))
    return torch.stack(embs)
