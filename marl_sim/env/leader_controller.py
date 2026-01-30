import numpy as np
# 导入碰撞检测工具
from marl_sim.world.collision import circle_collides


class LeaderController:
    def __init__(self, v_max=1.0, w_max=1.5, hold_steps=20):
        self.v_max = v_max
        self.w_max = w_max
        self.hold_steps = hold_steps
        self._hold = 0
        self._v, self._w = 0.0, 0.0

        # 避障参数
        self.lookahead_time = 1.5  # 预判未来 0.5 秒的位置
        self.radius = 0.7 # 领航车的物理半径（应与 MultiCarEnv 配置一致）

    def reset(self):
        self._hold = 0
        self._v = 0.0
        self._w = 0.0

    def get_action(self, state, grid) -> np.ndarray:
        """
        增加参数：
        state: 领航者当前的状态对象 (包含 x, y, theta)
        grid: 地图对象 (OccupancyGrid)
        """
        # 1. 正常的随机动作生成逻辑
        if self._hold <= 0:
            self._v = np.random.uniform(0.6 * self.v_max, self.v_max)
            self._w = np.random.uniform(-self.w_max, self.w_max)
            self._hold = self.hold_steps

        # 2. 边界/碰撞预判逻辑
        # 计算如果不改变方向，未来 lookahead_time 后的位置
        dt = self.lookahead_time
        next_x = state.x + self._v * np.cos(state.theta) * dt
        next_y = state.y + self._v * np.sin(state.theta) * dt

        # 使用 marl_sim.world.collision 中的工具检测该点是否碰撞
        is_collision = circle_collides(grid, next_x, next_y, self.radius)

        if is_collision:
            # 发现危险：强制大角度转向并减速
            # print("dd")
            self._w = self.w_max if self._w >= 0 else -self.w_max
            self._v = 0.2 * self.v_max  # 减速慢行寻路
            self._hold = 5  # 强制执行避障动作一段时间
        else:
            self._hold -= 1

        return np.array([self._v, self._w], dtype=np.float32)