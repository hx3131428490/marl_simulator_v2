from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING # <--- 1. 引入 TYPE_CHECKING

import numpy as np

# from marl_sim.env.multi_car_env import MultiCarEnv
from marl_sim.types import ActionDict, StepResult

if TYPE_CHECKING:
    from marl_sim.env.multi_car_env import MultiCarEnv

@dataclass
class TrajectoryRecorder:
    """
    轨迹记录器：用于收集仿真数据 (Dataset Collection)。
    将一局游戏 (Episode) 中的所有状态、动作、奖励序列化为 NumPy 数组并保存。

    Data Shapes:
    - states: (T+1, N_agents, 3) [x, y, theta] (包含初始状态)
    - actions: (T, N_agents, 2) [v, w]
    - rewards: (T, N_agents)
    - goals:   (N_agents, 2) [gx, gy] (静态目标)
    """
    n_agents: int

    def __post_init__(self) -> None:
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        # 初始化内部存储列表
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[np.ndarray] = []
        self._collided: List[np.ndarray] = []
        self._reached: List[np.ndarray] = []
        self._done: List[np.ndarray] = []
        self._step_count: List[int] = []
        self._goals: Optional[np.ndarray] = None
        self._meta: Dict[str, Any] = {}

    def reset_episode(self, env: MultiCarEnv) -> None:
        """
        开始新的一局记录。必须在 env.reset() 之后立即调用。
        记录初始状态 S_0 和静态目标。
        """
        self._states = [self._pack_states(env)]  # 记录 t=0
        self._actions = []
        self._rewards = []
        self._collided = []
        self._reached = []
        self._done = []
        self._step_count = []
        self._goals = self._pack_goals(env)
        # 保存环境配置作为元数据，方便追溯实验设置
        self._meta = {"config": asdict(env.cfg)}

    def record_step(self, env: MultiCarEnv, actions: ActionDict, out: StepResult) -> None:
        """
        记录单步交互数据。应在 env.step() 之后调用。
        记录: Action_t, Reward_t, Done_t, State_{t+1}
        """
        # 预分配 numpy 数组以提升性能
        a = np.zeros((self.n_agents, 2), dtype=np.float32)
        r = np.zeros((self.n_agents,), dtype=np.float32)
        c = np.zeros((self.n_agents,), dtype=np.bool_)
        rc = np.zeros((self.n_agents,), dtype=np.bool_)
        d = np.zeros((self.n_agents,), dtype=np.bool_)

        for i in range(self.n_agents):
            ai = actions[i]
            a[i, 0] = float(ai[0])
            a[i, 1] = float(ai[1])

            r[i] = float(out.reward[i])
            d[i] = bool(out.done[i])

            c[i] = bool(out.info["collided"][i])
            rc[i] = bool(out.info["reached"][i])

        self._actions.append(a)
        self._rewards.append(r)
        self._collided.append(c)
        self._reached.append(rc)
        self._done.append(d)
        self._step_count.append(int(out.info["step_count"]))

        # 记录更新后的状态 S_{t+1}
        self._states.append(self._pack_states(env))

    def to_npz_dict(self) -> Dict[str, Any]:
        """将列表堆叠 (Stack) 为紧凑的 NumPy 数组字典"""
        if self._goals is None or len(self._states) == 0:
            raise RuntimeError("TrajectoryRecorder has no episode data. Call reset_episode first.")

        states = np.stack(self._states, axis=0)  # Shape: (T+1, N, 3)
        goals = self._goals  # Shape: (N, 2)

        if len(self._actions) == 0:
            # 处理空数据的情况 (例如刚 reset 就结束)
            actions = np.zeros((0, self.n_agents, 2), dtype=np.float32)
            rewards = np.zeros((0, self.n_agents), dtype=np.float32)
            collided = np.zeros((0, self.n_agents), dtype=np.bool_)
            reached = np.zeros((0, self.n_agents), dtype=np.bool_)
            done = np.zeros((0, self.n_agents), dtype=np.bool_)
            step_count = np.zeros((0,), dtype=np.int32)
        else:
            actions = np.stack(self._actions, axis=0)
            rewards = np.stack(self._rewards, axis=0)
            collided = np.stack(self._collided, axis=0)
            reached = np.stack(self._reached, axis=0)
            done = np.stack(self._done, axis=0)
            step_count = np.asarray(self._step_count, dtype=np.int32)

        meta = self._meta
        return {
            "states": states,
            "goals": goals,
            "actions": actions,
            "rewards": rewards,
            "collided": collided,
            "reached": reached,
            "done": done,
            "step_count": step_count,
            "meta": np.asarray([repr(meta)], dtype=object),  # 由于npz不支持直接存dict,转为repr字符串
        }

    def save_npz(self, path: str | Path) -> Path:
        """保存为 .npz 压缩文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez_compressed 可以大幅减少磁盘占用
        np.savez_compressed(path, **self.to_npz_dict())
        return path

    def _pack_states(self, env: MultiCarEnv) -> np.ndarray:
        """辅助函数：将对象状态转为数组 [x, y, theta]"""
        s = np.zeros((self.n_agents, 3), dtype=np.float32)
        for i in range(self.n_agents):
            st = env.states[i]
            s[i, 0] = float(st.x)
            s[i, 1] = float(st.y)
            s[i, 2] = float(st.theta)
        return s

    def _pack_goals(self, env: MultiCarEnv) -> np.ndarray:
        """辅助函数：将目标转为数组 [gx, gy]"""
        g = np.zeros((self.n_agents, 2), dtype=np.float32)
        for i in range(self.n_agents):
            gx, gy = env.goals[i]
            g[i, 0] = float(gx)
            g[i, 1] = float(gy)
        return g