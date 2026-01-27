from __future__ import annotations

from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from marl_sim.world.grid_map import OccupancyGrid

Cell = Tuple[int, int]


def _bresenham_cells(ix0: int, iy0: int, ix1: int, iy1: int) -> Iterator[Cell]:
    """
    Bresenham 直线算法：生成两点间直线路径经过的所有网格坐标。
    使用纯整数运算，效率高且稳定。
    Yields: (ix, iy) 网格坐标
    """
    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    # 确定步进方向
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1

    x = ix0
    y = iy0

    # 根据直线斜率决定主轴 (x轴为主还是y轴为主)
    if dx >= dy:
        err = dx // 2
        while True:
            yield (x, y)
            if x == ix1 and y == iy1:
                break
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while True:
            yield (x, y)
            if x == ix1 and y == iy1:
                break
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy


def line_of_sight(
        grid: OccupancyGrid,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        treat_out_of_bounds_blocked: bool = True,
        include_endpoints: bool = True,
        ignore_start_cell: bool = True,
) -> bool:
    """
    检查两点之间的视线是否通畅 (Ray Casting)。

    Args:
        x0, y0: 起点物理坐标 (米)。
        x1, y1: 终点物理坐标 (米)。
        ignore_start_cell: 是否忽略起点所在的格子 (防止自己挡住视线)。
        treat_out_of_bounds_blocked: 视线射出地图外是否视为被阻挡。

    Returns:
        True: 视线通畅 (无障碍物)。
        False: 视线被阻挡。
    """
    # 1. 将物理坐标转换为网格索引
    # strict=False 允许坐标在地图外，后续在遍历时处理
    cell0 = grid.world_to_cell(x0, y0, strict=False)
    cell1 = grid.world_to_cell(x1, y1, strict=False)

    # 如果任一点无效(极远)，且认为出界即阻挡，则直接返回 False
    if treat_out_of_bounds_blocked and (cell0 is None or cell1 is None):
        return False

    # 如果两个点都在地图外，且不严格检查，可能需要更复杂的裁剪逻辑
    # 这里做简单处理：只要坐标能转成整数就跑 Bresenham
    # 实际工程中通常会先做线段与矩形的裁剪 (Cohen-Sutherland算法)
    if cell0 is None or cell1 is None:
        # 简化处理：若无法获取网格索引，视为不通
        return False

    ix0, iy0 = cell0
    ix1, iy1 = cell1

    # 2. 遍历路径上的每一个格子
    cells = _bresenham_cells(ix0, iy0, ix1, iy1)

    first = True
    for (ix, iy) in cells:
        # 处理起点忽略
        if first:
            first = False
            if ignore_start_cell:
                continue

        # 3. 检查当前格子状态
        # is_occupied_cell 会自动处理越界情况 (out_of_bounds_as_occupied)
        if grid.is_occupied_cell(ix, iy, out_of_bounds_as_occupied=treat_out_of_bounds_blocked):
            return False  # 碰到障碍物，视线阻断

    return True