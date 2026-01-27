from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from marl_sim.world.grid_map import OccupancyGrid


@dataclass(frozen=True)
class CollisionResult:
    x: float
    y: float
    collided: bool


def _circle_intersects_aabb(
        cx: float, cy: float, r: float,
        x0: float, y0: float, x1: float, y1: float
) -> bool:
    """
    底层几何检测：判断圆形是否与轴对齐矩形(AABB)相交。
    原理：寻找矩形上距离圆心最近的点 P，判断 dist(C, P) <= r。
    """
    # 1. 将圆心坐标 clamp 到矩形范围内，得到最近点 P(px, py)
    px = cx if x0 <= cx <= x1 else (x0 if cx < x0 else x1)
    py = cy if y0 <= cy <= y1 else (y0 if cy < y0 else y1)

    # 2. 计算圆心到 P 的距离平方 (避免开方运算以提升性能)
    dx = cx - px
    dy = cy - py
    return (dx * dx + dy * dy) <= (r * r)


def circle_collides(
        grid: OccupancyGrid,
        x: float,
        y: float,
        radius: float,
        *,
        out_of_bounds_as_occupied: bool = True
) -> bool:
    """
    检测：给定位置和半径的圆形，是否与地图上的障碍物发生碰撞。
    优化：只检测圆体包围盒覆盖的那几个网格，而非全图扫描。
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not np.isfinite([x, y, radius]).all():
        raise ValueError("circle_collides received non-finite values")

    ox, oy = grid.origin_world
    res = grid.resolution

    # 1. 快速边界检查 (空气墙)
    if out_of_bounds_as_occupied:
        if x - radius < ox or y - radius < oy:
            return True
        if x + radius > ox + grid.width_m or y + radius > oy + grid.height_m:
            return True

    # 2. 计算圆形覆盖的网格索引范围 (Candidate Cells)
    min_x = x - radius
    max_x = x + radius
    min_y = y - radius
    max_y = y + radius

    ix0 = int(np.floor((min_x - ox) / res))
    ix1 = int(np.floor((max_x - ox) / res))
    iy0 = int(np.floor((min_y - oy) / res))
    iy1 = int(np.floor((max_y - oy) / res))

    # 限制在地图索引范围内
    ix0 = max(ix0, 0)
    iy0 = max(iy0, 0)
    ix1 = min(ix1, grid.width_cells - 1)
    iy1 = min(iy1, grid.height_cells - 1)

    # 如果圆完全在地图外且允许越界，则不算碰撞
    if ix0 > ix1 or iy0 > iy1:
        return False

    # 3. 精确检测候选网格
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            # 只检测被占用的格子
            if grid.occ[iy, ix] == 0:
                continue

            # 计算该网格的物理坐标 AABB
            x0 = ox + ix * res
            y0 = oy + iy * res
            x1 = x0 + res
            y1 = y0 + res

            if _circle_intersects_aabb(x, y, radius, x0, y0, x1, y1):
                return True

    return False


def resolve_translation_with_collision(
        grid: OccupancyGrid,
        x_old: float,
        y_old: float,
        x_new: float,
        y_new: float,
        radius: float,
        *,
        mode: str = "backtrack",
        backtrack_iters: int = 12,
        out_of_bounds_as_occupied: bool = True
) -> CollisionResult:
    """
    响应：计算从旧位置移动到新位置的最终结果。
    如果发生碰撞，根据 mode 决定如何处理。

    Args:
        mode:
            "reject": 发生碰撞则直接返回旧位置 (原地不动)。
            "backtrack": 使用二分查找，寻找路径上离墙最近的安全点 (推荐)。
    """
    if mode not in {"reject", "backtrack"}:
        raise ValueError("mode must be 'reject' or 'backtrack'")

    # 1. 如果起点就已经撞墙了 (通常是因为初始化错误或被其他物体挤进去)，直接返回
    old_collides = circle_collides(
        grid, x_old, y_old, radius, out_of_bounds_as_occupied=out_of_bounds_as_occupied
    )
    if old_collides:
        return CollisionResult(x=x_old, y=y_old, collided=True)

    # 2. 如果终点是安全的，直接移动过去
    new_collides = circle_collides(
        grid, x_new, y_new, radius, out_of_bounds_as_occupied=out_of_bounds_as_occupied
    )
    if not new_collides:
        return CollisionResult(x=x_new, y=y_new, collided=False)

    # 3. 处理碰撞
    if mode == "reject":
        return CollisionResult(x=x_old, y=y_old, collided=True)

    # === Backtrack Mode (二分查找) ===
    # 在 [0, 1] 之间寻找最大的 t，使得 old + t * move 仍然安全
    dx = x_new - x_old
    dy = y_new - y_old

    lo = 0.0  # 已知安全 (old)
    hi = 1.0  # 已知碰撞 (new)

    for _ in range(backtrack_iters):
        mid = 0.5 * (lo + hi)
        xm = x_old + mid * dx
        ym = y_old + mid * dy

        if circle_collides(grid, xm, ym, radius, out_of_bounds_as_occupied=out_of_bounds_as_occupied):
            hi = mid  # 中点撞了，说明碰撞点在前半段
        else:
            lo = mid  # 中点安全，说明可以走得更远

    # 计算最终安全位置
    xs = x_old + lo * dx
    ys = y_old + lo * dy

    # 防御性检查：万一浮点误差导致算出来的点还是微弱接触
    if circle_collides(grid, xs, ys, radius, out_of_bounds_as_occupied=out_of_bounds_as_occupied):
        return CollisionResult(x=x_old, y=y_old, collided=True)

    return CollisionResult(x=xs, y=ys, collided=True)