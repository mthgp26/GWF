"""评估：K-means + Rand-Index（聚类）、SVM 10 折 CV（分类）。对应 SPEC §8。"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def kmeans_ri(embeddings, labels, n_clusters: int, seed: int = 0,
              n_init: int = 10) -> float:
    """K-means + Rand-Index（未调整 RI，与论文口径一致）。[SPEC §8.1]"""
    X = _as_numpy(embeddings)
    y = _as_numpy(labels).astype(int)
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit(X)
    return float(rand_score(y, km.labels_))


def svm_10fold(embeddings, labels, C_grid=None, seed: int = 0,
               kernel: str = "rbf") -> tuple[float, float]:
    """SVM 10 折 CV，C 在训练折内交叉验证选取（防泄漏）。[SPEC §8.2]

    Returns:
        ``(mean_acc, std_acc)``（折间 std）。
    """
    if C_grid is None:
        C_grid = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    X = _as_numpy(embeddings)
    y = _as_numpy(labels).astype(int)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        # 内层 CV（训练折内）选 C
        inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        best_C, best_score = C_grid[0], -1.0
        for C in C_grid:
            clf = SVC(C=C, kernel=kernel, gamma="scale")
            scores = cross_val_score(clf, Xtr, ytr, cv=inner, n_jobs=1)
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_C = C
        clf = SVC(C=best_C, kernel=kernel, gamma="scale")
        clf.fit(Xtr, ytr)
        accs.append(float(clf.score(Xte, yte)))
    return float(np.mean(accs)), float(np.std(accs))
