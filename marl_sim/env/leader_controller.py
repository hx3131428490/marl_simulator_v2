import numpy as np


class LeaderController:
    def __init__(self, v_max=1.0, w_max=1.5, hold_steps=20):
        self.v_max = v_max
        self.w_max = w_max
        self.hold_steps = hold_steps
        self._hold = 0
        self._v, self._w = 0.0, 0.0

    def reset(self):
        """
        重置控制器内部状态。
        将 _hold 设为 0 确保在 reset 后的第一个 step 会生成新的随机动作。
        """
        self._hold = 0
        self._v = 0.0
        self._w = 0.0

    def get_action(self) -> np.ndarray:
        """简单的脚本控制：随机改变速度并保持一段时间"""
        if self._hold <= 0:
            # 随机生成速度
            self._v = np.random.uniform(0.5 * self.v_max, self.v_max)
            self._w = np.random.uniform(-self.w_max, self.w_max)
            self._hold = self.hold_steps

        self._hold -= 1
        return np.array([self._v, self._w], dtype=np.float32)