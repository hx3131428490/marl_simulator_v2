from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from marl_sim.agent.state import CarState
from marl_sim.world.grid_map import OccupancyGrid
# 依赖上一节解读的视线检测函数
from marl_sim.world.visibility import line_of_sight


@dataclass(frozen=True)
class CommResult:
    """通信计算结果容器"""
    adj: np.ndarray  # 邻接矩阵 (N, N), True表示连通
    edges: List[Tuple[int, int]]  # 稀疏边列表 [(i, j), ...] (i < j)


def compute_adjacency(
        states: List[CarState],
        grid: OccupancyGrid,
        *,
        range_m: float,
        require_los: bool,
        enabled: bool = True,
) -> CommResult:
    """
    计算当前时刻所有智能体的通信拓扑图。

    Args:
        range_m: 最大通信半径 (米)。
        require_los: 是否需要视线通畅 (Line of Sight)。
                     如果为 True，且两车之间有障碍物，则无法通信。
    """
    n = len(states)
    # 初始化全 False 矩阵
    adj = np.zeros((n, n), dtype=bool)
    edges: List[Tuple[int, int]] = []

    # 如果通信功能被全局禁用，直接返回空图
    if not enabled:
        return CommResult(adj=adj, edges=edges)

    if range_m <= 0:
        raise ValueError("range_m must be positive")

    # 遍历所有唯一的对 (Pairs)
    # 复杂度 O(N^2)，适合 N < 100 的规模
    for i in range(n):
        for j in range(i + 1, n):
            si = states[i]
            sj = states[j]

            # 1. 距离检查 (欧氏距离)
            d = float(np.hypot(si.x - sj.x, si.y - sj.y))
            if d > range_m:
                continue  # 距离太远，跳过

            # 2. 视线检查 (Raycast)
            if require_los:
                # 调用 Bresenham 算法检查连线是否穿过障碍物
                # treat_out_of_bounds_blocked=True 表示地图外的虚空也阻挡信号
                ok = line_of_sight(grid, si.x, si.y, sj.x, sj.y, treat_out_of_bounds_blocked=True)
                if not ok:
                    continue  # 被墙挡住，跳过

            # 3. 建立连接 (无向图)
            adj[i, j] = True
            adj[j, i] = True
            edges.append((i, j))

    return CommResult(adj=adj, edges=edges)