"""工具：种子、计时、JSON 落盘、设备解析。对应 SPEC §12。"""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """统一设置随机种子（python/numpy/torch），保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | torch.device) -> torch.device:
    """解析设备；'mps' 不可用/不适用时退回 'cpu'。"""
    d = torch.device(device)
    if d.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if d.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return d


@contextmanager
def timer(name: str = ""):
    """计时上下文管理器。用法：``with timer('gwb'): ...``。"""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if name:
        print(f"[timer] {name}: {dt:.3f}s")


def save_json(path: str, data: Any) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)
