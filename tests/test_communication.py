import numpy as np

from marl_sim.agent.state import CarState
from marl_sim.world.grid_map import OccupancyGrid
from marl_sim.world.obstacles import clear_occupancy, fill_rect, RectObstacle
from marl_sim.env.communication import compute_adjacency


def test_adjacency_distance_only() -> None:
    g = OccupancyGrid(width_cells=20, height_cells=20, resolution=1.0)
    clear_occupancy(g)

    states = [
        CarState(x=2.5, y=2.5, theta=0.0),
        CarState(x=6.0, y=2.5, theta=0.0),
        CarState(x=15.0, y=2.5, theta=0.0),
    ]

    comm = compute_adjacency(states, g, range_m=5.0, require_los=False, enabled=True)
    assert comm.adj[0, 1] == True
    assert comm.adj[0, 2] == False
    assert comm.adj[1, 2] == False


def test_adjacency_with_los_blocked() -> None:
    g = OccupancyGrid(width_cells=20, height_cells=20, resolution=1.0)
    clear_occupancy(g)

    fill_rect(g, RectObstacle(ix0=5, iy0=2, w=1, h=1), value=True)

    states = [
        CarState(x=2.5, y=2.5, theta=0.0),
        CarState(x=7.5, y=2.5, theta=0.0),
    ]

    comm1 = compute_adjacency(states, g, range_m=10.0, require_los=False, enabled=True)
    assert comm1.adj[0, 1] == True

    comm2 = compute_adjacency(states, g, range_m=10.0, require_los=True, enabled=True)
    assert comm2.adj[0, 1] == False
