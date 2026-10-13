"""Phase 2 数据加载与统计校验测试（SPEC §3、prompt §9 Phase 2）。"""

import pytest
import torch

from gwf.data import (
    PAPER_STATS,
    check_dataset_stats,
    load_dataset,
    preprocess_graphs,
)


# 论文表 7.1 的 (数据集, 图数, 类别数, 属性维度)
DATASETS = [
    ("MUTAG", 188, 2, 7),
    ("PROTEIN", 1113, 2, 1),
    ("ENZYMES", 600, 6, 18),
    ("AIDS", 2000, 2, 4),
    ("PTC-MR", 344, 2, 18),
    ("IMDB-B", 1000, 2, 0),
    ("IMDB-M", 1500, 3, 0),
]


@pytest.mark.parametrize("name,n_graphs,n_classes,attr_dim", DATASETS)
def test_load_and_stats(name, n_graphs, n_classes, attr_dim):
    graphs, y = load_dataset(name, root="data")
    assert len(graphs) == n_graphs
    assert y.shape == (n_graphs,)
    assert len({g.y for g in graphs}) == n_classes
    if attr_dim > 0:
        assert all(g.F is not None and g.F.shape[1] == attr_dim for g in graphs)
    else:
        assert all(g.F is None for g in graphs)
    # 统计校验不抛异常（±0.5%）
    r = check_dataset_stats(name, graphs)
    assert r["ok"] is True


def test_mutag_edges_match_paper():
    # 关键口径：avg_edges = 唯一无向边 / 图数 = 19.79（不是 A.txt 行数/图数 = 39.58）
    # MUTAG_A.txt 共 7442 行 = 3721 条唯一无向边 × 2，故唯一边总数应为 3721。
    graphs, _ = load_dataset("MUTAG", root="data")
    total_edges = sum(int((g.C > 0).sum().item()) // 2 for g in graphs)
    assert total_edges == 3721


def test_adjacency_symmetric_and_zero_diag():
    graphs, _ = load_dataset("MUTAG", root="data")
    for g in graphs[:20]:
        C = g.C
        assert torch.allclose(C, C.t()), "邻接矩阵应对称"
        assert torch.all(C.diagonal() == 0), "对角应为 0"


def test_preprocess_normalize_and_diag():
    graphs, _ = load_dataset("MUTAG", root="data")
    out = preprocess_graphs(graphs[:10], normalize=True, zero_diag=True)
    for g in out:
        assert g.C.max().item() <= 1.0 + 1e-6
        assert torch.all(g.C.diagonal() == 0)


def test_unknown_dataset_raises():
    with pytest.raises((KeyError, FileNotFoundError)):
        load_dataset("NOT_A_DATASET", root="data")
