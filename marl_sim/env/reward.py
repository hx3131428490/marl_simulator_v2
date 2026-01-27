from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from marl_sim.agent.state import CarState


def compute_rewards(
        states: List[CarState],
        goals: List[Tuple[float, float]],
        collided_flags: List[bool],
        *,
        distance_weight: float,
        collision_penalty: float,
) -> Dict[int, float]:
    """
    计算所有智能体在当前步的奖励。
    策略: 距离目标越近分越高(负数逼近0)，发生碰撞则扣分。

    Args:
        states: 所有智能体的当前状态列表。
        goals: 所有智能体的目标坐标列表 [(gx, gy), ...]。
        collided_flags: 对应智能体本回合是否发生碰撞。
        distance_weight: 距离权重的系数 (通常设为 1.0)。
        collision_penalty: 碰撞惩罚的绝对值 (通常设为 1.0 或更大)。

    Returns:
        Dict[int, float]: 映射 AgentID -> Reward。
    """
    rewards: Dict[int, float] = {}

    for i, s in enumerate(states):
        # 1. 计算与目标的欧氏距离
        gx, gy = goals[i]
        d = float(np.hypot(s.x - gx, s.y - gy))

        # 2. 基础奖励：距离越远惩罚越大 (Dense Reward)
        # 这种设计鼓励智能体走直线最短路径
        r = -distance_weight * d

        # 3. 碰撞惩罚：如果撞墙或撞人，额外扣分
        if collided_flags[i]:
            r -= collision_penalty

        rewards[i] = float(r)

    return rewards