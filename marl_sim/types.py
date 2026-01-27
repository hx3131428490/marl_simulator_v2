from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

# === 基础类型定义 ===
# 统一类型别名，方便后续维护和静态检查
AgentID = int
Vec2 = Tuple[float, float]

Action = np.ndarray
ActionDict = Dict[AgentID, Action]  # 动作字典: {ID: 动作数组}

Obs = np.ndarray
ObsDict = Dict[AgentID, Obs]  # 观测字典: {ID: 观测数组}

RewardDict = Dict[AgentID, float]  # 奖励字典: {ID: 得分}
DoneDict = Dict[AgentID, bool]  # 结束字典: {ID: 是否结束}
InfoDict = Dict[str, Any]  # 额外信息


@dataclass(frozen=True)
class StepResult:
    """
    环境单步交互的返回结果容器。
    frozen=True 表示实例化后不可修改，防止数据意外被篡改。
    """
    obs: ObsDict
    reward: RewardDict
    done: DoneDict
    info: InfoDict


def assert_action_dict(actions: Mapping[AgentID, np.ndarray], n_agents: int) -> None:
    """
    动作数据的防御性校验。在传入环境前确保数据格式和安全性。
    """
    # 1. 检查收到动作的数量是否对应智能体总数
    if len(actions) != n_agents:
        raise ValueError(f"ActionDict size mismatch: got {len(actions)} expected {n_agents}")

    for aid, act in actions.items():
        # 2. 检查 ID 类型 (必须是整数)
        if not isinstance(aid, int):
            raise TypeError(f"AgentID must be int, got {type(aid)}")

        # 3. 检查动作数据类型 (必须是 NumPy 数组)
        if not isinstance(act, np.ndarray):
            raise TypeError(f"Action must be np.ndarray, got {type(act)}")

        # 4. 关键约束: 检查动作维度 (限制为 2D 向量)
        if act.shape != (2,):
            raise ValueError(f"Action shape must be (2,), got {act.shape}")

        # 5. 数值安全检查 (防止 NaN 或 Inf 导致物理引擎崩溃)
        if not np.isfinite(act).all():
            raise ValueError("Action contains non-finite values")