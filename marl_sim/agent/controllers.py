from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING

from marl_sim.types import ActionDict
# from marl_sim.env.multi_car_env import MultiCarEnv
if TYPE_CHECKING:
    from marl_sim.env.multi_car_env import MultiCarEnv


def wrap_to_pi(theta: float) -> float:
    """角度归一化辅助函数"""
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


class Controller:
    """
    控制器基类 (策略接口)。
    所有的 Agent 策略（规则、随机、神经网络）都应继承此类。
    """
    def reset(self, env: MultiCarEnv) -> None:
        """在环境 reset 时调用，用于重置控制器内部状态 (如有)"""
        return None

    def act(self, env: MultiCarEnv) -> ActionDict:
        """
        核心决策方法。
        Args:
            env: 仿真环境实例 (通常用于获取 states 和 goals)
        Returns:
            ActionDict: 所有智能体的动作字典 {agent_id: action}
        """
        raise NotImplementedError


@dataclass
class RandomController(Controller):
    """
    随机控制器。
    产生完全随机的动作，用于测试环境稳定性和作为最差基准。
    """
    rng: np.random.Generator
    v_max: float
    w_max: float

    def act(self, env: MultiCarEnv) -> ActionDict:
        actions: ActionDict = {}
        for i in range(env.n_agents):
            # 均匀分布随机采样
            v = float(self.rng.uniform(-self.v_max, self.v_max))
            w = float(self.rng.uniform(-self.w_max, self.w_max))
            actions[i] = np.array([v, w], dtype=np.float64)
        return actions


@dataclass
class GoalSeekingController(Controller):
    """
    基于规则的目标追踪控制器 (Heuristic Policy)。
    使用 P-Controller 进行导航，不具备避障能力。
    """
    k_w: float = 2.0  # 转向比例系数

    def act(self, env: MultiCarEnv) -> ActionDict:
        v_max = float(env.cfg.agent.v_max)
        w_max = float(env.cfg.agent.w_max)

        actions: ActionDict = {}
        for i in range(env.n_agents):
            s = env.states[i]
            gx, gy = env.goals[i]

            # 1. 计算目标方位
            dx = float(gx - s.x)
            dy = float(gy - s.y)

            desired = float(np.arctan2(dy, dx))
            err = wrap_to_pi(desired - float(s.theta))

            # 2. 计算角速度 (P控制)
            w = float(np.clip(self.k_w * err, -w_max, w_max))

            # 3. 计算线速度 (角度误差越大，速度越慢)
            v_scale = max(0.0, 1.0 - abs(err) / np.pi)
            v = float(v_max * v_scale)

            actions[i] = np.array([v, w], dtype=np.float64)

        return actions