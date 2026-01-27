from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from marl_sim.world.grid_map import OccupancyGrid


@dataclass(frozen=True)
class RectObstacle:
    """矩形障碍物的数据结构，定义位置和尺寸"""
    ix0: int  # 左侧索引 (包含)
    iy0: int  # 底部索引 (包含)
    w: int  # 宽度 (格子数)
    h: int  # 高度 (格子数)

    @property
    def ix1(self) -> int:
        """右侧索引 (不包含)，用于 slice 切片"""
        return self.ix0 + self.w

    @property
    def iy1(self) -> int:
        """顶部索引 (不包含)，用于 slice 切片"""
        return self.iy0 + self.h

    @property
    def area(self) -> int:
        return self.w * self.h


def clear_occupancy(grid: OccupancyGrid) -> None:
    """清空地图，重置为全 0 (无障碍)"""
    grid.occ.fill(0)


def fill_rect(grid: OccupancyGrid, rect: RectObstacle, value: bool = True) -> None:
    """
    在地图上绘制一个矩形障碍物。
    利用 NumPy 切片操作实现高效赋值。
    """
    # 基础参数校验
    if rect.w <= 0 or rect.h <= 0:
        raise ValueError("RectObstacle width and height must be positive")
    if rect.ix0 < 0 or rect.iy0 < 0:
        raise ValueError("RectObstacle origin must be non-negative")
    # 确保不超出地图边界
    if rect.ix1 > grid.width_cells or rect.iy1 > grid.height_cells:
        raise ValueError("RectObstacle exceeds grid bounds")

    # [y_start:y_end, x_start:x_end] = 1
    grid.occ[rect.iy0:rect.iy1, rect.ix0:rect.ix1] = 1 if value else 0


def occupancy_ratio(grid: OccupancyGrid) -> float:
    """计算地图的占用率 (0.0 ~ 1.0)，用于评估地图复杂度"""
    return float(grid.occ.mean())


def generate_random_rectangles(
        grid: OccupancyGrid,
        rng: np.random.Generator,
        n_rect: int,
        *,
        w_range: Tuple[int, int] = (3, 7),
        h_range: Tuple[int, int] = (3, 7),
        margin_cells: int = 1,
        allow_overlap: bool = False,
        max_tries: int = 20000,
        write_to_grid: bool = True,
) -> List[RectObstacle]:
    """
    随机生成多个矩形障碍物并放置在地图上。
    使用拒绝采样 (Rejection Sampling) 来避免重叠。

    Args:
        margin_cells: 距离地图边缘保留的空隙格数，防止封死边界。
        allow_overlap: 是否允许新生成的矩形覆盖已有的障碍物。
                       如果为 False，生成时会检测碰撞，若有重叠则重试。
        write_to_grid: 是否立即将生成的矩形“画”到 grid.occ 数组上。
                       如果为 False，只返回矩形对象但不修改地图。
    """
    if n_rect < 0:
        raise ValueError("n_rect must be non-negative")
    # ... (省略参数校验代码以节省篇幅，逻辑同上) ...
    if w_range[0] <= 0 or h_range[0] <= 0:
        raise ValueError("w_range and h_range must be positive")
    if w_range[0] > w_range[1] or h_range[0] > h_range[1]:
        raise ValueError("invalid range")
    if margin_cells < 0:
        raise ValueError("margin_cells must be non-negative")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive")

    rects: List[RectObstacle] = []
    placed = 0
    tries = 0

    min_w, max_w = w_range
    min_h, max_h = h_range

    while placed < n_rect and tries < max_tries:
        tries += 1

        # 1. 随机生成尺寸
        w = int(rng.integers(min_w, max_w + 1))
        h = int(rng.integers(min_h, max_h + 1))

        # 2. 计算可放置的有效范围 (考虑边距)
        ix_min = margin_cells
        iy_min = margin_cells
        ix_max = grid.width_cells - margin_cells - w
        iy_max = grid.height_cells - margin_cells - h

        if ix_max < ix_min or iy_max < iy_min:
            raise ValueError("grid too small for given rectangle size range and margin")

        # 3. 随机生成位置
        ix0 = int(rng.integers(ix_min, ix_max + 1))
        iy0 = int(rng.integers(iy_min, iy_max + 1))

        rect = RectObstacle(ix0=ix0, iy0=iy0, w=w, h=h)

        # 4. 碰撞检测 (如果禁止重叠)
        if not allow_overlap:
            # 提取该区域的切片
            region = grid.occ[rect.iy0:rect.iy1, rect.ix0:rect.ix1]
            # 如果切片里已经有障碍物 (非0值)，则放弃本次生成，重试
            if np.any(region != 0):
                continue

        # 5. 放置成功
        rects.append(rect)
        placed += 1

        # 如果需要，立即写入地图数组 (这会影响下一次迭代的碰撞检测)
        if write_to_grid:
            fill_rect(grid, rect, value=True)

    # 如果尝试了 max_tries 次仍未放满，抛出错误提示用户减少矩形数量或增大地图
    if placed != n_rect:
        raise RuntimeError(f"Failed to place all rectangles. placed={placed}, requested={n_rect}, tries={tries}")

    return rects