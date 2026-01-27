from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from marl_sim.agent.state import CarState, wrap_to_pi
from marl_sim.world.grid_map import OccupancyGrid
from marl_sim.world.collision import resolve_translation_with_collision


@dataclass(frozen=True)
class KinematicStepResult:
    """物理步进的返回结果"""
    state: CarState  # 更新后的状态
    collided: bool  # 是否在移动过程中发生了碰撞
    v_cmd: float  # 实际执行的线速度 (经限幅后)
    w_cmd: float  # 实际执行的角速度 (经限幅后)


def clip_unicycle_action(v: float, w: float, v_max: float, w_max: float) -> Tuple[float, float]:
    """
    将动作指令限制在物理可行范围内 [-max, max]。
    模拟电机的最大功率限制。
    """
    if v_max <= 0 or w_max <= 0:
        raise ValueError("v_max and w_max must be positive")
    if not np.isfinite([v, w, v_max, w_max]).all():
        raise ValueError("action or limits are not finite")

    v_c = float(np.clip(v, -v_max, v_max))
    w_c = float(np.clip(w, -w_max, w_max))
    return v_c, w_c


def unicycle_integrate(state: CarState, v: float, w: float, dt: float) -> Tuple[float, float, float]:
    """
    独轮车(Unicycle)模型的运动学积分 (Euler Integration)。
    x' = x + v * cos(theta) * dt
    y' = y + v * sin(theta) * dt
    th' = theta + w * dt
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    state.validate()

    x, y, th = state.x, state.y, state.theta
    # 计算位移增量
    x_new = float(x + v * np.cos(th) * dt)
    y_new = float(y + v * np.sin(th) * dt)
    # 计算角度增量并归一化
    th_new = wrap_to_pi(float(th + w * dt))

    return x_new, y_new, th_new


def step_unicycle(
        state: CarState,
        action: np.ndarray,
        *,
        dt: float,
        v_max: float,
        w_max: float,
        radius: float,
        grid: Optional[OccupancyGrid] = None,
        collision_mode: str = "backtrack",
        backtrack_iters: int = 12,
) -> KinematicStepResult:
    """
    执行单步物理仿真：包含动作限幅、运动积分和碰撞处理。

    Args:
        action: numpy数组 [v, w]，单位通常是 m/s 和 rad/s。
        grid: 如果提供地图，将启用碰撞检测与响应。
        collision_mode: 碰撞响应模式 ('reject' 或 'backtrack')。

    Returns:
        KinematicStepResult: 包含新状态和碰撞信息。
    """
    # 1. 基础校验
    if not isinstance(action, np.ndarray):
        raise TypeError("action must be a numpy array")
    if action.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {action.shape}")
    if radius <= 0:
        raise ValueError("radius must be positive")

    v_raw = float(action[0])
    w_raw = float(action[1])

    # 2. 动作限幅 (模拟电机特性)
    v_cmd, w_cmd = clip_unicycle_action(v_raw, w_raw, v_max, w_max)

    # 3. 运动学积分 (计算理想情况下的下一时刻位置)
    x_new, y_new, th_new = unicycle_integrate(state, v_cmd, w_cmd, dt)

    # 4. 物理交互 (碰撞检测与修正)
    collided = False
    if grid is not None:
        # 调用物理引擎：检查从 (x_old, y_old) 到 (x_new, y_new) 是否撞墙
        # 注意：这里假设原地旋转不会导致碰撞 (圆形底盘特性)，只检测位移碰撞
        res = resolve_translation_with_collision(
            grid=grid,
            x_old=float(state.x),
            y_old=float(state.y),
            x_new=float(x_new),
            y_new=float(y_new),
            radius=float(radius),
            mode=collision_mode,
            backtrack_iters=backtrack_iters,
            out_of_bounds_as_occupied=True,
        )
        # 更新为修正后的位置 (如果没撞，就是原计划位置；如果撞了，就是墙前的位置)
        x_new, y_new = float(res.x), float(res.y)
        collided = bool(res.collided)

    # 5. 组装结果
    new_state = CarState(x=x_new, y=y_new, theta=th_new)
    return KinematicStepResult(state=new_state, collided=collided, v_cmd=v_cmd, w_cmd=w_cmd)