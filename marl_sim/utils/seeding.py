from __future__ import annotations

import random
import numpy as np


def seed_everything(seed: int, deterministic_numpy: bool = True) -> np.random.Generator:
    """
    初始化全局随机种子，确保实验的可复现性。

    Args:
        seed: 随机种子数值。
        deterministic_numpy: 是否创建一个确定性的 NumPy Generator。
                             如果为 False，则 Generator 会随机初始化（不可复现）。

    Returns:
        np.random.Generator: 一个独立的 NumPy 随机生成器实例（推荐在代码中传递使用此对象，而非调用全局 np.random）。
    """
    # 1. 设置 Python 标准库的种子
    random.seed(seed)

    # 2. 设置 NumPy 全局 (Legacy) 状态的种子
    # (为了兼容那些直接调用 np.random.xxx 的旧代码或第三方库)
    np.random.seed(seed)

    # 3. 创建并返回一个新的、局部的 Generator (NumPy 新版最佳实践)
    if deterministic_numpy:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()  # 随机初始化

    return rng