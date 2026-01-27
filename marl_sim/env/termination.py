from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from marl_sim.agent.state import CarState


def compute_reached(
        states: List[CarState],
        goals: List[Tuple[float, float]],
        *,
        goal_tolerance: float,
) -> List[bool]:
    """
    检查每个智能体是否进入了目标范围。

    Args:
        goal_tolerance: 判定到达的距离阈值 (半径)。

    Returns:
        List[bool]: 对应每个智能体是否到达。
    """
    reached: List[bool] = []
    for i, s in enumerate(states):
        gx, gy = goals[i]
        # 计算欧氏距离
        d = float(np.hypot(s.x - gx, s.y - gy))
        reached.append(d <= goal_tolerance)
    return reached


def compute_done_dict(
        reached: List[bool],
        *,
        step_count: int,
        max_steps: int,
) -> Dict[int, bool]:
    """
    计算环境的结束状态 (Done Flags)。

    逻辑:
    1. 时间耗尽 (超时) -> 全员结束。
    2. 所有智能体都到达目标 (all reached) -> 全员结束 (合作成功)。

    注意: 这是一个'全局同步结束'的机制。
    """
    # 只要满足超时 或 全员到达，游戏就结束
    global_done = (step_count >= max_steps) or all(reached)

    # 将这个全局状态广播给所有智能体
    return {i: bool(global_done) for i in range(len(reached))}