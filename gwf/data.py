"""TUDataset 原始格式加载与统计校验。

对应 SPEC §3（数据规格）、prompt.md §8。

解析规则（TUDataset 原始文本格式，来自 https://www.chrsmrrs.com/graphkerneldatasets/）：
- ``<NAME>_graph_indicator.txt`` : 每行一个全局节点索引对应的图 id（1 起）。
- ``<NAME>_A.txt``               : 每行 ``u, v``（1 起，逗号分隔）。无向图数据集中每条
                                    无向边会被存成 ``(u,v)`` 与 ``(v,u)`` 两条记录，因此
                                    直接按记录累加即可得到对称邻接矩阵。
- ``<NAME>_graph_labels.txt``    : 每行一个图的类别标签。
- ``<NAME>_node_attributes.txt`` : 每行一个节点的浮点属性（逗号分隔）；若不存在则回退到
                                    ``<NAME>_node_labels.txt`` 的离散标签做 one-hot。
- ``<NAME>_node_labels.txt``     : 每行一个节点的离散标签（categorical）。

统计口径（与论文表 7.1 对齐，见 SPEC §3.1）：
- ``avg_nodes`` = 总节点数 / 图数。
- ``avg_edges`` = 总**唯一无向边**数 / 图数（``C.sum()/2``，因邻接矩阵对称且无自环）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    """单张图。对应 prompt.md §8.1 的 ``Graph`` 结构。

    - ``C``: 邻接矩阵 ``(N, N)`` float32，元素 0/1（原始），无自环、对称。
    - ``F``: 节点特征 ``(N, D)`` float32；无节点属性时为 ``None``（IMDB-B/M）。
    - ``y``: 类别标签 int（0 起）。
    """

    C: torch.Tensor
    F: Optional[torch.Tensor]
    y: int
    name: str = field(default="")


# ---------------------------------------------------------------------------
# 名称映射与论文统计（SPEC §3.1 / prompt §5.3）
# ---------------------------------------------------------------------------
# 论文名 / 别名 -> TUDataset 原始目录名
_NAME_ALIASES = {
    "AIDS": "AIDS",
    "ENZYMES": "ENZYMES",
    "MUTAG": "MUTAG",
    "PROTEIN": "PROTEINS",
    "PROTEINS": "PROTEINS",
    "PROTEIN-F": "PROTEINS_full",
    "PROTEINS-F": "PROTEINS_full",
    "PROTEINS_full": "PROTEINS_full",
    "PTC-MR": "PTC_MR",
    "PTC_MR": "PTC_MR",
    "IMDB-B": "IMDB-BINARY",
    "IMDB-BINARY": "IMDB-BINARY",
    "IMDB-M": "IMDB-MULTI",
    "IMDB-MULTI": "IMDB-MULTI",
}

# 论文表 7.1 统计（校验期望值）
PAPER_STATS = {
    "AIDS":       dict(n_graphs=2000, n_classes=2, attr_dim=4,  avg_nodes=15.69, avg_edges=16.20),
    "PROTEIN":    dict(n_graphs=1113, n_classes=2, attr_dim=1,  avg_nodes=39.06, avg_edges=72.82),
    "PROTEIN-F":  dict(n_graphs=1113, n_classes=2, attr_dim=29, avg_nodes=39.06, avg_edges=72.82),
    "ENZYMES":    dict(n_graphs=600,  n_classes=6, attr_dim=18, avg_nodes=32.63, avg_edges=62.14),
    "MUTAG":      dict(n_graphs=188,  n_classes=2, attr_dim=7,  avg_nodes=17.93, avg_edges=19.79),
    "PTC-MR":     dict(n_graphs=344,  n_classes=2, attr_dim=18, avg_nodes=14.29, avg_edges=14.69),
    "IMDB-B":     dict(n_graphs=1000, n_classes=2, attr_dim=0,  avg_nodes=19.77, avg_edges=96.53),
    "IMDB-M":     dict(n_graphs=1500, n_classes=3, attr_dim=0,  avg_nodes=13.00, avg_edges=65.94),
}

_ATTR_DIM_OVERRIDE = {"PROTEIN-F": 29, "PROTEIN": 1}  # 论文对 PROTEIN-F 用 29 维完整属性


def _resolve_dir(name: str) -> str:
    key = name.upper().replace("_", "-").replace("-", "-")
    if name in _NAME_ALIASES:
        return _NAME_ALIASES[name]
    # 大小写不敏感兜底
    for k, v in _NAME_ALIASES.items():
        if k.upper() == name.upper():
            return v
    raise KeyError(f"未知数据集名 '{name}'，可用：{sorted(set(_NAME_ALIASES.values()))}")


def _read_lines(path: str) -> list[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() != ""]


def _parse_float_row(line: str) -> list[float]:
    return [float(x) for x in line.replace(",", " ").split()]


# ---------------------------------------------------------------------------
# 主加载函数
# ---------------------------------------------------------------------------
def load_dataset(name: str, root: str = "data") -> tuple[list[Graph], torch.LongTensor]:
    """加载 TUDataset 数据集。

    Args:
        name: 论文名或 TUDataset 名（见 ``_NAME_ALIASES``）。
        root: 数据根目录，其下需存在 ``<dir>/<NAME>_*.txt``。

    Returns:
        ``(graphs, y)``：``graphs`` 为 ``list[Graph]``；``y`` 为 ``(num_graphs,)``
        int64 类别标签（0 起）。
    """
    raw = _resolve_dir(name)
    d = os.path.join(root, raw)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"数据目录不存在：{d}（请先下载 {raw}.zip 并解压到 {root}/）")

    indicator = [int(x) for x in _read_lines(os.path.join(d, f"{raw}_graph_indicator.txt"))]
    graph_labels = [int(x) for x in _read_lines(os.path.join(d, f"{raw}_graph_labels.txt"))]
    edges = _read_lines(os.path.join(d, f"{raw}_A.txt"))

    n_nodes = len(indicator)
    # 图 id 1 起 -> 0 起
    gid = np.array(indicator, dtype=np.int64) - 1
    n_graphs = int(gid.max()) + 1

    # 每个图的节点集合与局部索引
    node_local = [None] * n_nodes
    counts = np.bincount(gid, minlength=n_graphs)
    node_list = [np.empty(c, dtype=np.int64) for c in counts]  # 每图全局节点 id
    fill = np.zeros(n_graphs, dtype=np.int64)
    for node in range(n_nodes):
        g = int(gid[node])
        idx = int(fill[g])
        node_list[g][idx] = node
        node_local[node] = (g, idx)
        fill[g] += 1

    # 邻接矩阵：按 A.txt 累加（无向图每条边出现两次，得到对称矩阵）
    Cs = [np.zeros((c, c), dtype=np.float32) for c in counts]
    for line in edges:
        u_s, v_s = line.split(",")
        u, v = int(u_s) - 1, int(v_s) - 1
        g_u, i = node_local[u]
        g_v, j = node_local[v]
        if g_u != g_v:
            # TUDataset 保证同图；出现跨图说明解析有误
            raise RuntimeError(f"{raw}: 边 ({u+1},{v+1}) 跨图（{g_u} vs {g_v}）")
        Cs[g_u][i, j] += 1.0

    # 节点特征：优先 node_attributes，其次 node_labels 的 one-hot，否则 None
    attr_path = os.path.join(d, f"{raw}_node_attributes.txt")
    lbl_path = os.path.join(d, f"{raw}_node_labels.txt")
    attr_dim = PAPER_STATS[name]["attr_dim"]

    Fs = None
    if os.path.isfile(attr_path):
        rows = _read_lines(attr_path)
        Fs = [np.zeros((c, 0), dtype=np.float32) for c in counts]
        # 两遍：先定维度，再填
        sample = _parse_float_row(rows[0])
        D = len(sample)
        Fs = [np.zeros((c, D), dtype=np.float32) for c in counts]
        for node, line in enumerate(rows):
            g, idx = node_local[node]
            Fs[g][idx] = np.asarray(_parse_float_row(line), dtype=np.float32)
    elif os.path.isfile(lbl_path):
        rows = _read_lines(lbl_path)
        labels = np.array([int(x) for x in rows], dtype=np.int64)
        n_cat = int(labels.max()) + 1
        Fs = [np.zeros((c, n_cat), dtype=np.float32) for c in counts]
        for node in range(n_nodes):
            g, idx = node_local[node]
            Fs[g][idx, labels[node]] = 1.0

    # 类别标签映射到 0 起
    uniq = np.unique(np.array(graph_labels))
    lab_map = {int(u): i for i, u in enumerate(uniq)}
    ys = torch.tensor([lab_map[l] for l in graph_labels], dtype=torch.long)

    graphs = [
        Graph(C=torch.from_numpy(Cs[g]), F=None if Fs is None else torch.from_numpy(Fs[g]),
              y=int(ys[g]), name=name)
        for g in range(n_graphs)
    ]
    return graphs, ys


# ---------------------------------------------------------------------------
# 统计校验（SPEC §3.4）
# ---------------------------------------------------------------------------
def _stats(graphs: list[Graph]) -> dict:
    n = len(graphs)
    total_nodes = sum(int(g.C.shape[0]) for g in graphs)
    # 唯一无向边 = 对称邻接矩阵非零元 / 2（无自环）
    total_edges = sum(int((g.C > 0).sum().item()) // 2 for g in graphs)
    n_classes = len({g.y for g in graphs})
    attr_dim = graphs[0].F.shape[1] if graphs[0].F is not None else 0
    return dict(n_graphs=n, n_classes=n_classes, attr_dim=attr_dim,
                avg_nodes=total_nodes / n, avg_edges=total_edges / n)


def check_dataset_stats(name: str, graphs: list[Graph], tol: float = 0.005) -> dict:
    """将解析结果与论文表 7.1 逐列比对（±0.5%），不一致抛异常。

    返回比对字典，字段：``ok``、``paper``、``actual``、``diffs``。
    """
    paper = PAPER_STATS[name]
    actual = _stats(graphs)
    diffs = {}
    for k in ("n_graphs", "n_classes", "avg_nodes", "avg_edges"):
        p, a = paper[k], actual[k]
        if p == 0:
            diffs[k] = dict(paper=p, actual=a, rel=float(a == 0), ok=(a == 0))
            continue
        rel = abs(a - p) / abs(p)
        diffs[k] = dict(paper=p, actual=a, rel=rel, ok=(rel <= tol))
    ok = all(d["ok"] for d in diffs.values())
    if not ok:
        bad = {k: v for k, v in diffs.items() if not v["ok"]}
        raise AssertionError(
            f"[{name}] 统计与论文表 7.1 不一致（±0.5% 容差）：{bad}"
        )
    return dict(ok=ok, paper=paper, actual=actual, diffs=diffs)


def preprocess_graphs(
    graphs: list[Graph],
    symmetrize: bool = False,
    normalize: bool = True,
    zero_diag: bool = True,
) -> list[Graph]:
    """预处理（SPEC §3.3）：对称化（可选）、归一化到 [0,1]、对角置 0。

    注意：不修改输入，返回新 ``Graph`` 列表。默认 ``symmetrize=False``
    （原始数据已对称），``normalize=True``，``zero_diag=True``。
    """
    out = []
    for g in graphs:
        C = g.C.clone()
        if symmetrize:
            C = torch.maximum(C, C.t())
        if normalize:
            mx = C.max()
            if mx is not None and mx.item() > 0:
                C = C / mx
        if zero_diag:
            C = C.clone()
            C.fill_diagonal_(0.0)
        out.append(Graph(C=C, F=None if g.F is None else g.F.clone(), y=g.y, name=g.name))
    return out
