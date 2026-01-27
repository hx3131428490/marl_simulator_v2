from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from marl_sim.types import ActionDict, ObsDict, StepResult, assert_action_dict
from marl_sim.config.schema import SimConfig
from marl_sim.utils.seeding import seed_everything

from marl_sim.agent.state import CarState
from marl_sim.agent.kinematics import step_unicycle

from marl_sim.world.grid_map import OccupancyGrid
from marl_sim.world.obstacles import clear_occupancy, generate_random_rectangles
from marl_sim.world.collision import circle_collides

from marl_sim.env.reward import compute_rewards
from marl_sim.env.termination import compute_reached, compute_done_dict

# [新增] 导入通信和观测模块
# 注意：根据之前的文件结构，compute_adjacency 在 marl_sim.world.comm
from marl_sim.env.communication import compute_adjacency
# 注意：根据之前的文件结构，global_observation 在 marl_sim.env.observation
from marl_sim.agent.observation import global_observation


def _sample_free_pose(
    grid: OccupancyGrid,
    rng: np.random.Generator,
    radius: float,
    existing_xy: List[Tuple[float, float]],
    *,
    min_sep: float,
    max_tries: int = 20000,
) -> Tuple[float, float]:
    """
    在地图上采样一个安全的、无碰撞的随机位置。
    1. 不在墙里 (circle_collides)
    2. 不与其他已存在的位置重叠 (existing_xy check)
    """
    ox, oy = grid.origin_world
    for _ in range(max_tries):
        # 1. 随机选一个格子中心
        ix = int(rng.integers(0, grid.width_cells))
        iy = int(rng.integers(0, grid.height_cells))
        xy = grid.cell_to_world(ix, iy, center=True, strict=False)
        if xy is None: continue
        x, y = xy

        # 2. 检查地形碰撞 (墙)
        if circle_collides(grid, x, y, radius, out_of_bounds_as_occupied=True):
            continue

        # 3. 检查实体间重叠 (人挤人)
        ok = True
        for (px, py) in existing_xy:
            if float(np.hypot(x - px, y - py)) < min_sep:
                ok = False
                break
        if not ok:
            continue

        return float(x), float(y)

    raise RuntimeError("Failed to sample a collision-free pose (map too crowded?)")


@dataclass
class MultiCarEnv:
    """
    多智能体小车仿真环境核心类。
    遵循 Gym/PettingZoo 风格接口: reset(), step()
    """
    cfg: SimConfig

    def __post_init__(self) -> None:
        """初始化组件"""
        self.cfg.validate()
        # 初始化随机数生成器
        self.rng: np.random.Generator = seed_everything(self.cfg.seed)
        # 初始化地图对象
        self.grid: OccupancyGrid = OccupancyGrid(
            width_cells=self.cfg.map.width_cells,
            height_cells=self.cfg.map.height_cells,
            resolution=self.cfg.map.resolution,
        )
        # 状态存储
        self.states: List[CarState] = []
        self.goals: List[Tuple[float, float]] = []
        self.step_count: int = 0
        # [新增] 初始化通信缓存字段
        self._last_comm = None

    @property
    def n_agents(self) -> int:
        return int(self.cfg.agent.n_agents)

    @property
    def obs_dim(self) -> int:
        # 每个智能体观测维度 = 5 * n_agents (全信息)
        return int(self.n_agents * 5)

    def reset(self, *, seed: Optional[int] = None) -> ObsDict:
        """重置环境：生成新地图，随机放置智能体和目标"""
        if seed is not None:
            self.rng = seed_everything(int(seed))
        self.step_count = 0

        # 1. 重置地图
        clear_occupancy(self.grid)
        if self.cfg.obstacles.enabled and self.cfg.obstacles.n_rect > 0:
            generate_random_rectangles(
                self.grid,
                self.rng,
                n_rect=int(self.cfg.obstacles.n_rect),
                w_range=tuple(self.cfg.obstacles.w_range),
                h_range=tuple(self.cfg.obstacles.h_range),
                margin_cells=int(self.cfg.obstacles.margin_cells),
                allow_overlap=bool(self.cfg.obstacles.allow_overlap),
                max_tries=int(self.cfg.obstacles.max_tries),
                write_to_grid=True,
            )

        # 2. 随机放置智能体 (Start)
        self.states = []
        used_xy: List[Tuple[float, float]] = []
        min_sep = float(2.2 * self.cfg.agent.radius) # 保持一定间距

        for i in range(self.n_agents):
            x, y = _sample_free_pose(self.grid, self.rng, float(self.cfg.agent.radius), used_xy, min_sep=min_sep)
            theta = float(self.rng.uniform(-np.pi, np.pi))
            self.states.append(CarState(x=x, y=y, theta=theta))
            used_xy.append((x, y))

        # 3. 随机放置目标点 (Goal)
        self.goals = []
        used_goal_xy: List[Tuple[float, float]] = []
        for i in range(self.n_agents):
            gx, gy = _sample_free_pose(self.grid, self.rng, float(self.cfg.agent.radius), used_goal_xy, min_sep=min_sep)
            self.goals.append((gx, gy))
            used_goal_xy.append((gx, gy))

        # [新增] 4. 计算通信拓扑并生成初始观测
        comm = compute_adjacency(
            self.states,
            self.grid,
            range_m=float(self.cfg.comm.range_m),
            require_los=bool(self.cfg.comm.require_los),
            enabled=bool(self.cfg.comm.enabled),
        )
        self._last_comm = comm

        gobs = global_observation(self.states, self.goals)
        return {i: gobs.copy() for i in range(self.n_agents)}

    def step(self, actions: ActionDict) -> StepResult:
        """主循环：物理积分 -> 碰撞检测 -> 奖励计算 -> 状态更新"""
        assert_action_dict(actions, self.n_agents)

        collided_flags: List[bool] = [False for _ in range(self.n_agents)]
        new_states: List[CarState] = []

        # 1. 物理更新 (Physics Update)
        for i in range(self.n_agents):
            out = step_unicycle(
                state=self.states[i],
                action=actions[i],
                dt=float(self.cfg.dt),
                v_max=float(self.cfg.agent.v_max),
                w_max=float(self.cfg.agent.w_max),
                radius=float(self.cfg.agent.radius),
                grid=self.grid,
                collision_mode="backtrack",
                backtrack_iters=14,
            )
            new_states.append(out.state)
            collided_flags[i] = bool(out.collided)

        self.states = new_states
        self.step_count += 1

        # 2. 逻辑判定 (Game Logic)
        reached = compute_reached(self.states, self.goals, goal_tolerance=float(self.cfg.goal_tolerance))
        done = compute_done_dict(reached, step_count=int(self.step_count), max_steps=int(self.cfg.max_steps))
        reward = compute_rewards(
            self.states,
            self.goals,
            collided_flags,
            distance_weight=float(self.cfg.distance_weight),
            collision_penalty=float(self.cfg.collision_penalty),
        )

        # [新增] 3. 计算通信拓扑
        comm = compute_adjacency(
            self.states,
            self.grid,
            range_m=float(self.cfg.comm.range_m),
            require_los=bool(self.cfg.comm.require_los),
            enabled=bool(self.cfg.comm.enabled),
        )
        self._last_comm = comm

        # [修改] 4. 观测组装 (Observation)
        gobs = global_observation(self.states, self.goals)
        obs: ObsDict = {i: gobs.copy() for i in range(self.n_agents)}

        # [修改] 5. 调试信息 (Info) - 包含通信数据
        info: Dict[str, object] = {
            "step_count": int(self.step_count),
            "collided": {i: bool(collided_flags[i]) for i in range(self.n_agents)},
            "reached": {i: bool(reached[i]) for i in range(self.n_agents)},
            "adj": comm.adj,
            "edges": comm.edges,
        }

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def render(self) -> None:
        """(预留接口) 可视化渲染"""
        return None

    def close(self) -> None:
        """(预留接口) 资源释放"""
        return None