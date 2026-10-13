"""图聚类脚本：训练 GWF → K-means → Rand-Index，落盘结果 JSON。

用法：``python scripts/run_clustering.py --config configs/gwf_mutag.yaml``
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
from gwf.evaluate import kmeans_ri
from gwf.model import GWF
from gwf.train import train_gwf
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

    model = GWF(K, Nk, I, L=cfg["L"], M=cfg["M"], gamma=gamma, gwd=cfg["gwd"],
                inner_iter=cfg.get("inner_iter", 50),
                loss_normalize=cfg.get("loss_normalize", False))
    # 预热初始化（K 个随机图）
    indices = torch.randperm(I)[:K].tolist()
    model.warm_init_V(Cs, indices)

    t0 = time.time()
    losses = train_gwf(model, Cs, lr=cfg["lr"], epochs=cfg["epochs"],
                       optim=cfg["optim"], device="cpu")  # 见 NOTES §1：CPU 更快
    runtime = time.time() - t0

    emb = model.embed().detach()
    n_classes = len({g.y for g in graphs})
    ri = kmeans_ri(emb, labels, n_clusters=n_classes, seed=cfg["seed"])

    result = dict(
        dataset=cfg["dataset"], model=cfg["model"], K=K, M=cfg["M"], L=cfg["L"],
        gamma=gamma, lr=cfg["lr"], epochs=cfg["epochs"], optim=cfg["optim"],
        inner_iter=cfg.get("inner_iter", 50), seed=cfg["seed"],
        metric="RI", value=ri, final_loss=losses[-1], runtime_sec=runtime,
        device="cpu",
    )
    out_path = os.path.join(
        args.out, cfg["dataset"],
        f"{cfg['model']}_K{K}_{cfg['optim']}_seed{cfg['seed']}.json",
    )
    save_json(out_path, result)
    print(f"[{cfg['dataset']}] K={K} {cfg['optim']} seed={cfg['seed']} "
          f"RI={ri:.4f} loss={losses[-1]:.4f} time={runtime:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
