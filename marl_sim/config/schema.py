from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass
class MapConfig:
    """地图基础尺寸与分辨率配置"""
    width_cells: int = 80
    height_cells: int = 60
    resolution: float = 0.1

    def validate(self) -> None:
        if self.width_cells <= 0 or self.height_cells <= 0:
            raise ValueError("MapConfig width_cells and height_cells must be positive")
        if self.resolution <= 0:
            raise ValueError("MapConfig resolution must be positive")


@dataclass
class AgentConfig:
    """智能体物理属性配置"""
    n_agents: int = 4
    radius: float = 0.25
    v_max: float = 1.0
    w_max: float = 1.5

    def validate(self) -> None:
        if self.n_agents <= 0:
            raise ValueError("AgentConfig n_agents must be positive")
        if self.radius <= 0:
            raise ValueError("AgentConfig radius must be positive")
        if self.v_max <= 0 or self.w_max <= 0:
            raise ValueError("AgentConfig v_max and w_max must be positive")


@dataclass
class RenderConfig:
    """可视化渲染配置"""
    enabled: bool = True
    fps: int = 60
    cell_px: int = 12

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("RenderConfig fps must be positive")
        if self.cell_px <= 0:
            raise ValueError("RenderConfig cell_px must be positive")


@dataclass
class CommConfig:
    """通信功能配置"""
    enabled: bool = True
    range_m: float = 6.0
    require_los: bool = True

    def validate(self) -> None:
        if self.range_m <= 0:
            raise ValueError("CommConfig range_m must be positive")


@dataclass
class ObstacleConfig:
    """
    [新增] 随机障碍物生成配置
    控制地图生成的难易程度
    """
    enabled: bool = False        # 是否启用随机障碍物
    n_rect: int = 10            # 障碍物数量
    w_range: Tuple[int, int] = (3, 7)  # 宽度随机范围 (min, max)
    h_range: Tuple[int, int] = (3, 7)  # 高度随机范围 (min, max)
    margin_cells: int = 1       # 边缘留空距离
    allow_overlap: bool = False # 是否允许重叠
    max_tries: int = 20000      # 生成算法的最大尝试次数

    def validate(self) -> None:
        if self.n_rect < 0:
            raise ValueError("ObstacleConfig n_rect must be non-negative")
        if self.w_range[0] <= 0 or self.h_range[0] <= 0:
            raise ValueError("ObstacleConfig ranges must be positive")
        if self.w_range[0] > self.w_range[1] or self.h_range[0] > self.h_range[1]:
            raise ValueError("ObstacleConfig invalid range ordering")
        if self.margin_cells < 0:
            raise ValueError("ObstacleConfig margin_cells must be non-negative")
        if self.max_tries <= 0:
            raise ValueError("ObstacleConfig max_tries must be positive")


@dataclass
class SimConfig:
    """
    仿真总配置
    包含全局参数、奖励参数及所有子模块配置
    """
    seed: int = 0
    dt: float = 0.05
    max_steps: int = 500

    # [新增] 强化学习奖励函数相关参数
    goal_tolerance: float = 0.5    # 到达目标的判定距离 (米)
    # collision_penalty: float = 1.0 # 发生碰撞时的扣分 (绝对值)
    distance_weight: float = 1.0   # 距离奖励的权重系数

    # 子模块配置
    map: MapConfig = field(default_factory=MapConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig) # [新增]

    formation_weight: float = 2.0  # 编队位置误差权重
    yaw_weight: float = 1.0  # 航向一致性权重
    collision_penalty: float = 10.0  # 碰撞惩罚

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError("SimConfig dt must be positive")
        if self.max_steps <= 0:
            raise ValueError("SimConfig max_steps must be positive")
        # 校验奖励参数
        if self.goal_tolerance <= 0:
            raise ValueError("SimConfig goal_tolerance must be positive")
        if self.collision_penalty < 0:
            raise ValueError("SimConfig collision_penalty must be non-negative")
        if self.distance_weight < 0:
            raise ValueError("SimConfig distance_weight must be non-negative")

        # 级联校验子配置
        self.map.validate()
        self.agent.validate()
        self.render.validate()
        self.comm.validate()
        self.obstacles.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimConfig":
        """从字典加载配置，支持嵌套结构"""
        def sub(name: str, klass):
            if name not in data or data[name] is None:
                return klass()
            if not isinstance(data[name], Mapping):
                raise TypeError(f"{name} must be a mapping")
            return klass(**dict(data[name]))

        cfg = cls(
            seed=int(data.get("seed", 0)),
            dt=float(data.get("dt", 0.05)),
            max_steps=int(data.get("max_steps", 500)),
            # 读取新增的顶层参数
            goal_tolerance=float(data.get("goal_tolerance", 0.5)),
            collision_penalty=float(data.get("collision_penalty", 1.0)),
            distance_weight=float(data.get("distance_weight", 1.0)),
            # 读取子模块
            map=sub("map", MapConfig),
            agent=sub("agent", AgentConfig),
            render=sub("render", RenderConfig),
            comm=sub("comm", CommConfig),
            obstacles=sub("obstacles", ObstacleConfig),
        )
        cfg.validate()
        return cfg