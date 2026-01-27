from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from marl_sim.agent.state import CarState


def global_observation(states: List[CarState], goals: List[Tuple[float, float]]) -> np.ndarray:
    """
    生成全局观测向量 (Global State)。
    通常用于 Centralized Critic。
    格式: [x1, y1, th1, gx1, gy1, x2, y2, th2, gx2, gy2, ...] 平铺。
    """
    vec: List[float] = []
    for i in range(len(states)):
        s = states[i]
        gx, gy = goals[i]
        # 记录绝对坐标
        vec.extend([float(s.x), float(s.y), float(s.theta), float(gx), float(gy)])
    return np.asarray(vec, dtype=np.float32)


def local_observation_placeholder(
        states: List[CarState],
        goals: List[Tuple[float, float]],
        adj: np.ndarray,
        *,
        max_neighbors: int = 3,
) -> Dict[int, np.ndarray]:
    """
    生成局部观测向量 (Local Observation)。
    用于 Decentralized Actor。

    Args:
        adj: 通信邻接矩阵 (来自 comm.py)，决定了谁能被看到。
        max_neighbors: 最大关注邻居数。用于保证输出向量长度固定。

    Returns:
        Dict[agent_id, obs_vector]
    """
    n = len(states)
    out: Dict[int, np.ndarray] = {}

    for i in range(n):
        s = states[i]
        gx, gy = goals[i]

        # 1. 自身目标相对信息 (Relative Goal)
        dxg = float(gx - s.x)
        dyg = float(gy - s.y)

        # 自身基础特征: [绝对位置, 绝对朝向, 相对目标位移]
        # 注: 有些策略会移除绝对位置 x,y 以实现完全的位置无关性
        feats: List[float] = [float(s.x), float(s.y), float(s.theta), dxg, dyg]

        # 2. 筛选邻居
        # 根据 adj 矩阵找到通信范围内的邻居索引
        neigh = [j for j in range(n) if j != i and bool(adj[i, j])]
        # 截断: 只保留前 k 个 (通常按距离排序会更好，这里简单按索引)
        neigh = neigh[:max_neighbors]

        # 3. 邻居相对信息 (Relative Neighbor Info)
        for j in neigh:
            sj = states[j]
            # 记录邻居相对于我的位置和朝向差
            feats.extend([float(sj.x - s.x), float(sj.y - s.y), float(sj.theta - s.theta)])

        # 4. 填充 (Padding)
        # 如果邻居数量不足 max_neighbors，用 0 补齐，保持向量长度一致
        need = max_neighbors - len(neigh)
        for _ in range(need):
            feats.extend([0.0, 0.0, 0.0])

        out[i] = np.asarray(feats, dtype=np.float32)

    return out