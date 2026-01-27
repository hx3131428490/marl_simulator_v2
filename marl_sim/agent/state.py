from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def wrap_to_pi(theta: float) -> float:
    """
    将任意角度归一化到 [-pi, pi) 区间。
    处理角度周期性问题的标准数学工具。
    例如: 3*pi -> pi; -pi -> -pi
    """
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class CarState:
    """
    定义智能体的物理状态 (SE2 Group: x, y, theta)。
    """
    x: float
    y: float
    theta: float  # 弧度 (radians)

    def validate(self) -> None:
        """检查数值有效性 (防止 NaN/Inf 污染仿真)"""
        if not np.isfinite([self.x, self.y, self.theta]).all():
            raise ValueError("CarState contains non-finite values")

    def normalized(self) -> "CarState":
        """
        返回一个经过角度归一化的新状态对象。
        在物理积分后通常需要调用此方法。
        """
        self.validate()
        return CarState(x=float(self.x), y=float(self.y), theta=wrap_to_pi(float(self.theta)))

    def as_tuple(self) -> Tuple[float, float, float]:
        """转换为元组，方便解包或作为观测输出"""
        return float(self.x), float(self.y), float(self.theta)