import numpy as np

from marl_sim.world.grid_map import OccupancyGrid
from marl_sim.world.obstacles import clear_occupancy, fill_rect, RectObstacle
from marl_sim.world.visibility import line_of_sight


def test_line_of_sight_clear() -> None:
    g = OccupancyGrid(width_cells=20, height_cells=20, resolution=1.0)
    clear_occupancy(g)

    assert line_of_sight(g, 2.5, 2.5, 15.5, 2.5) is True
    assert line_of_sight(g, 2.5, 2.5, 15.5, 15.5) is True


def test_line_of_sight_blocked_by_obstacle() -> None:
    g = OccupancyGrid(width_cells=20, height_cells=20, resolution=1.0)
    clear_occupancy(g)

    fill_rect(g, RectObstacle(ix0=8, iy0=10, w=1, h=1), value=True)

    assert line_of_sight(g, 2.5, 10.5, 15.5, 10.5) is False
