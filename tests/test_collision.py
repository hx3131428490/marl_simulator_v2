import numpy as np
import pytest

from marl_sim.world.grid_map import OccupancyGrid
from marl_sim.world.obstacles import clear_occupancy, fill_rect, RectObstacle
from marl_sim.world.collision import circle_collides, resolve_translation_with_collision


def test_circle_boundary_collision() -> None:
    """
    测试地图边缘的碰撞检测 (空气墙)。
    """
    g = OccupancyGrid(width_cells=10, height_cells=10, resolution=1.0)
    clear_occupancy(g)

    # 1. 在地图中心：安全
    assert circle_collides(g, x=5.0, y=5.0, radius=0.4) is False

    # 2. 压线或出界：应判定为碰撞 (当 out_of_bounds_as_occupied=True)
    # 左边界检测 (x=0.2, r=0.25 -> left edge at -0.05 < 0)
    assert circle_collides(g, x=0.2, y=5.0, radius=0.25, out_of_bounds_as_occupied=True) is True
    # 右、下、上边界同理
    assert circle_collides(g, x=9.8, y=5.0, radius=0.25, out_of_bounds_as_occupied=True) is True
    assert circle_collides(g, x=5.0, y=0.2, radius=0.25, out_of_bounds_as_occupied=True) is True
    assert circle_collides(g, x=5.0, y=9.8, radius=0.25, out_of_bounds_as_occupied=True) is True

    # 3. 如果允许出界 (False)，则不碰撞
    assert circle_collides(g, x=0.2, y=5.0, radius=0.25, out_of_bounds_as_occupied=False) is False


def test_circle_cell_collision_single_block() -> None:
    """
    测试与单个障碍物块的几何碰撞精度。
    """
    g = OccupancyGrid(width_cells=10, height_cells=10, resolution=1.0)
    clear_occupancy(g)

    # 在 (3,4) 放一个障碍物
    fill_rect(g, RectObstacle(ix0=3, iy0=4, w=1, h=1), value=True)

    # 1. 圆心直接在障碍物里：必撞
    x_c, y_c = g.cell_to_world(3, 4, center=True)
    assert circle_collides(g, x=x_c, y=y_c, radius=0.2) is True

    # 2. 离得远：不撞
    assert circle_collides(g, x=1.0, y=1.0, radius=0.2) is False

    # 3. 临界测试：贴得很近但没碰到
    # 障碍物左边缘在 x=3.0。圆心在 x=2.6，半径=0.3 -> 右缘到 2.9。没碰到。
    assert circle_collides(g, x=2.6, y=4.5, radius=0.3) is False


def test_resolve_backtrack_produces_non_colliding_point() -> None:
    """
    核心测试：验证 'backtrack' 模式能否正确计算出碰撞前的最近安全点。
    """
    g = OccupancyGrid(width_cells=20, height_cells=20, resolution=1.0)
    clear_occupancy(g)

    # 在中间放一堵墙
    fill_rect(g, RectObstacle(ix0=9, iy0=9, w=3, h=3), value=True)

    radius = 0.4
    x_old, y_old = 5.0, 10.0 # 安全起点
    x_new, y_new = 12.0, 10.0 # 穿墙终点

    # 确保前提条件正确
    assert circle_collides(g, x_old, y_old, radius) is False
    assert circle_collides(g, x_new, y_new, radius) is True

    # 执行回溯计算
    res = resolve_translation_with_collision(
        g, x_old, y_old, x_new, y_new, radius, mode="backtrack", backtrack_iters=14
    )

    # 验证结果
    assert res.collided is True
    # 算出来的结果必须是安全的！(不能卡进墙里)
    assert circle_collides(g, res.x, res.y, radius) is False

    # 算出来的点必须在路径上
    assert x_old <= res.x <= x_new
    assert abs(res.y - y_old) < 1e-6


def test_resolve_reject_returns_old() -> None:
    """
    测试 'reject' 模式：一旦碰撞，直接返回原点。
    """
    g = OccupancyGrid(width_cells=20, height_cells=20, resolution=1.0)
    clear_occupancy(g)
    fill_rect(g, RectObstacle(ix0=9, iy0=9, w=3, h=3), value=True)

    radius = 0.4
    x_old, y_old = 5.0, 10.0
    x_new, y_new = 12.0, 10.0

    res = resolve_translation_with_collision(
        g, x_old, y_old, x_new, y_new, radius, mode="reject"
    )
    assert res.collided is True
    # 必须原地不动
    assert res.x == x_old and res.y == y_old