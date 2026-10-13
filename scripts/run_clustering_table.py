"""批量运行聚类并汇总为 markdown 表（Phase 6）。

用法：``python scripts/run_clustering_table.py``（内部定义待跑配置列表）。
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gwf.data import PAPER_STATS, load_dataset, preprocess_graphs
from gwf.evaluate import kmeans_ri
from gwf.model import GWF
from gwf.train import train_gwf
from gwf.utils import set_seed

# 待跑配置：(dataset, gwd, K, seeds)
RUNS = [
    ("MUTAG", "ppa", 15, [0, 1, 2]),
    ("MUTAG", "badmm", 15, [0, 1, 2]),
    ("ENZYMES", "ppa", 15, [0, 1, 2]),
    ("ENZYMES", "badmm", 15, [0, 1, 2]),
    ("PROTEIN", "ppa", 15, [0, 1, 2]),
    ("PROTEIN", "badmm", 15, [0, 1, 2]),
    ("IMDB-B", "ppa", 15, [0, 1, 2]),
    ("IMDB-B", "badmm", 15, [0, 1, 2]),
]

PAPER_RI = {  # 论文表 7.3 GWF_PPA / GWF_BADMM（%）
    "AIDS": (99.0, 98.4), "PROTEIN": (59.8, 60.0), "ENZYMES": (72.3, 71.9),
    "MUTAG": (69.2, 68.8), "PTC-MR": (52.5, 52.4), "IMDB-B": (51.8, 51.3),
    "IMDB-M": (55.9, 55.1),
}


def run_one(dataset, gwd, K, seed):
    set_seed(seed)
    torch.manual_seed(seed)
    graphs, labels = load_dataset(dataset, root="data")
    graphs = preprocess_graphs(graphs, normalize=True, zero_diag=True)
    Cs = [g.C for g in graphs]
    I = len(graphs)
    Nk = 10
    gamma = 0.01 if gwd == "ppa" else 1.0
    model = GWF(K, Nk, I, L=5, M=20, gamma=gamma, gwd=gwd, inner_iter=50)
    model.warm_init_V(Cs, torch.randperm(I)[:K].tolist())
    train_gwf(model, Cs, lr=0.01, epochs=10, optim="alternating", device="cpu")
    emb = model.embed().detach()
    n_classes = len({g.y for g in graphs})
    return kmeans_ri(emb, labels, n_clusters=n_classes, seed=seed)


def main():
    rows = []
    for dataset, gwd, K, seeds in RUNS:
        vals = [run_one(dataset, gwd, K, s) for s in seeds]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        paper = PAPER_RI[dataset][0 if gwd == "ppa" else 1]
        rows.append((dataset, gwd, K, mean, std, paper))
        print(f"{dataset:9s} {gwd:6s} K={K}: RI={mean:.3f}±{std:.3f} (paper {paper})", flush=True)

    print("\n| 数据集 | 模型 | K | 复现 RI(%) | 论文 RI(%) | 偏差 |")
    print("|---|---|---|---|---|---|")
    for dataset, gwd, K, mean, std, paper in rows:
        print(f"| {dataset} | GWF_{gwd} | {K} | {mean*100:.1f}±{std*100:.1f} | {paper} | {mean*100-paper:+.1f} |")


if __name__ == "__main__":
    main()
