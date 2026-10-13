"""图分类脚本：训练 GWF/GWF-GNN → SVM 10 折 CV，落盘结果 JSON。

用法：``python scripts/run_classification.py --config configs/gwf_gnn_enzymes.yaml``
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gwf.data import load_dataset, preprocess_graphs
from gwf.evaluate import svm_10fold
from gwf.model import GWF, GINEncoder, GWFGNN
from gwf.train import embed_gnn, train_gwf, train_gwf_gnn
from gwf.utils import save_json, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    graphs, labels = load_dataset(cfg["dataset"], root=args.data_root)
    graphs = preprocess_graphs(
        graphs,
        symmetrize=cfg.get("symmetrize", False),
        normalize=cfg.get("normalize", True),
        zero_diag=cfg.get("zero_diag", True),
    )
    Cs = [g.C for g in graphs]
    I = len(graphs)
    K = cfg["K"]
    Nk = cfg["factor_nodes"]
    gamma = cfg["gamma_ppa"] if cfg["gwd"] == "ppa" else cfg["gamma_badmm"]
    inner_iter = cfg.get("inner_iter", 50)

    t0 = time.time()
    if cfg["model"] == "GWF":
        model = GWF(K, Nk, I, L=cfg["L"], M=cfg["M"], gamma=gamma, gwd=cfg["gwd"],
                    inner_iter=inner_iter, loss_normalize=cfg.get("loss_normalize", False))
        indices = torch.randperm(I)[:K].tolist()
        model.warm_init_V(Cs, indices)
        train_gwf(model, Cs, lr=cfg["lr"], epochs=cfg["epochs"],
                  optim=cfg["optim"], device="cpu")
        emb = model.embed().detach()
    else:  # GWF-GNN
        # 输入维度：节点属性维度，无属性用 1（全 1 或度）
        in_dim = graphs[0].F.shape[1] if graphs[0].F is not None else 1
        encoder = GINEncoder(in_dim, cfg.get("gnn_hidden", 32), K,
                             n_layers=cfg.get("gnn_layers", 5))
        model = GWFGNN(K, Nk, encoder, L=cfg["L"], M=cfg["M"], gamma=gamma,
                       gwd=cfg["gwd"], inner_iter=inner_iter,
                       loss_normalize=cfg.get("loss_normalize", False))
        indices = torch.randperm(I)[:K].tolist()
        model.warm_init_V(Cs, indices)
        train_gwf_gnn(model, graphs, lr=cfg["lr"], epochs=cfg["epochs"], device="cpu")
        emb = embed_gnn(model, graphs, device="cpu")
    runtime = time.time() - t0

    acc, std = svm_10fold(emb, labels, seed=cfg["seed"], kernel=cfg.get("svm_kernel", "rbf"))

    result = dict(
        dataset=cfg["dataset"], model=cfg["model"], K=K, M=cfg["M"], L=cfg["L"],
        gamma=gamma, lr=cfg["lr"], epochs=cfg["epochs"], optim=cfg.get("optim", "joint"),
        inner_iter=inner_iter, seed=cfg["seed"], metric="Accuracy", value=acc, std=std,
        runtime_sec=runtime, device="cpu",
    )
    out_path = os.path.join(args.out, cfg["dataset"],
                            f"{cfg['model']}_K{K}_seed{cfg['seed']}.json")
    save_json(out_path, result)
    print(f"[{cfg['dataset']}] {cfg['model']} K={K} seed={cfg['seed']} "
          f"Acc={acc:.2f}±{std:.2f} time={runtime:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
