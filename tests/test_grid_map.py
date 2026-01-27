import numpy as np
import pytest

from marl_sim.world.grid_map import OccupancyGrid


def test_cell_world_roundtrip_center() -> None:
    """
    测试坐标转换的一致性 (Round-trip test)。
    验证流程：网格索引 -> 世界坐标(中心点) -> 网格索引
    确保转换过程没有精度丢失或逻辑错误。
    """
    g = OccupancyGrid(width_cells=10, height_cells=8, resolution=0.5)
    rng = np.random.default_rng(0)

    for _ in range(200):
        ix = int(rng.integers(0, g.width_cells))
        iy = int(rng.integers(0, g.height_cells))

        # 转换去
        x, y = g.cell_to_world(ix, iy, center=True)
        # 转换回
        ix2, iy2 = g.world_to_cell(x, y)

        # 必须相等
        assert (ix, iy) == (ix2, iy2)


def test_world_to_cell_boundaries_left_closed_right_open() -> None:
    """
    测试地图边界条件的严格性。
    遵循 "左闭右开" [min, max) 原则。
    """
    g = OccupancyGrid(width_cells=4, height_cells=3, resolution=1.0)

    # 1. 测试左下角 (包含)
    assert g.world_to_cell(0.0, 0.0) == (0, 0)

    # 2. 测试右上角极值 (包含)
    eps = 1e-9
    assert g.world_to_cell(g.width_m - eps, g.height_m - eps) == (g.width_cells - 1, g.height_cells - 1)

    # 3. 测试正好压在右上边界线上 (应视为越界报错)
    with pytest.raises(ValueError):
        g.world_to_cell(g.width_m, 0.0)
    with pytest.raises(ValueError):
        g.world_to_cell(0.0, g.height_m)

    # 4. 测试负坐标 (应视为越界报错)
    with pytest.raises(ValueError):
        g.world_to_cell(-eps, 0.0)


def test_occupancy_set_and_query() -> None:
    """
    测试障碍物状态的读写，以及'空气墙'逻辑。
    """
    g = OccupancyGrid(width_cells=5, height_cells=5, resolution=0.2)

    # 测试状态翻转
    assert g.is_occupied_cell(2, 3) is False
    g.set_occupied_cell(2, 3, True)
    assert g.is_occupied_cell(2, 3) is True
    g.set_occupied_cell(2, 3, False)
    assert g.is_occupied_cell(2, 3) is False

    # 测试地图外区域的判定 (Ghost Wall)
    # 默认情况下，地图外被视为障碍物 (True)，防止智能体走出世界
    assert g.is_occupied_cell(-1, 0, out_of_bounds_as_occupied=True) is True
    # 如果显式允许忽略边界，则返回 False
    assert g.is_occupied_cell(-1, 0, out_of_bounds_as_occupied=False) is False