import numpy as np
import pytest

from marl_sim.world.grid_map import OccupancyGrid
from marl_sim.world.obstacles import clear_occupancy, generate_random_rectangles, occupancy_ratio


def test_random_rectangles_count_and_ratio_and_values() -> None:
    """
    基础功能测试：
    验证生成的矩形数量是否正确，地图数值是否合法(仅0和1)，以及占用率是否在合理区间。
    """
    g = OccupancyGrid(width_cells=50, height_cells=40, resolution=0.1)
    rng = np.random.default_rng(0)

    clear_occupancy(g)
    rects = generate_random_rectangles(
        g,
        rng,
        n_rect=10,
        w_range=(3, 7),
        h_range=(3, 7),
        margin_cells=1,
        allow_overlap=False,
        max_tries=20000,
        write_to_grid=True,
    )

    # 1. 数量检查
    assert len(rects) == 10

    # 2. 占用率检查 (防止生成空地图或全黑地图)
    r = occupancy_ratio(g)
    assert 0.05 <= r <= 0.25

    # 3. 数据完整性检查 (确保地图只包含 0 和 1)
    unique_vals = set(np.unique(g.occ).tolist())
    assert unique_vals.issubset({0, 1})


def test_random_rectangles_reproducible_with_seed() -> None:
    """
    可复现性测试 (Reproducibility Test)：
    验证给定相同的随机种子，生成的地图布局是否完全一致。
    这是强化学习环境稳定性的基石。
    """
    # 初始化两套完全一样的环境
    g1 = OccupancyGrid(width_cells=60, height_cells=45, resolution=0.1)
    g2 = OccupancyGrid(width_cells=60, height_cells=45, resolution=0.1)

    clear_occupancy(g1)
    clear_occupancy(g2)

    # 使用相同的种子初始化随机生成器
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    # 执行生成
    _ = generate_random_rectangles(
        g1, rng1, n_rect=12, w_range=(4, 9), h_range=(2, 6), margin_cells=2, allow_overlap=False
    )
    _ = generate_random_rectangles(
        g2, rng2, n_rect=12, w_range=(4, 9), h_range=(2, 6), margin_cells=2, allow_overlap=False
    )

    # 比较两个地图的 numpy 数组是否完全相等
    assert np.array_equal(g1.occ, g2.occ)