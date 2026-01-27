import numpy as np

from marl_sim.agent.state import CarState, wrap_to_pi
from marl_sim.agent.kinematics import clip_unicycle_action, step_unicycle
from marl_sim.world.grid_map import OccupancyGrid
from marl_sim.world.obstacles import clear_occupancy, fill_rect, RectObstacle


def test_wrap_to_pi_range() -> None:
    a = wrap_to_pi(3.5)   # > pi
    b = wrap_to_pi(-3.5)  # < -pi
    assert -np.pi <= a < np.pi
    assert -np.pi <= b < np.pi


def test_action_clipping() -> None:
    v, w = clip_unicycle_action(v=10.0, w=-10.0, v_max=1.2, w_max=2.0)
    assert abs(v - 1.2) < 1e-12
    assert abs(w + 2.0) < 1e-12


def test_unicycle_step_no_collision_deterministic() -> None:
    s = CarState(x=1.0, y=2.0, theta=0.0)
    action = np.array([1.0, 0.0], dtype=np.float64)

    out = step_unicycle(
        s,
        action,
        dt=0.1,
        v_max=2.0,
        w_max=2.0,
        radius=0.25,
        grid=None,
    )

    assert abs(out.state.x - 1.1) < 1e-12
    assert abs(out.state.y - 2.0) < 1e-12
    assert abs(out.state.theta - 0.0) < 1e-12
    assert out.collided is False


def test_unicycle_step_with_collision_backtrack() -> None:
    g = OccupancyGrid(width_cells=20, height_cells=20, resolution=1.0)
    clear_occupancy(g)

    # Place a vertical wall at x cell index 10, spanning y 0..19
    fill_rect(g, RectObstacle(ix0=10, iy0=0, w=1, h=20), value=True)

    s = CarState(x=8.0, y=10.0, theta=0.0)
    action = np.array([10.0, 0.0], dtype=np.float64)

    out = step_unicycle(
        s,
        action,
        dt=0.2,
        v_max=20.0,
        w_max=2.0,
        radius=0.4,
        grid=g,
        collision_mode="backtrack",
        backtrack_iters=16,
    )

    # Should collide and stop before the wall, so x must be < 10.0 - radius in world units
    assert out.collided is True
    assert out.state.x < 10.0 - 0.4 + 1e-6
    assert abs(out.state.y - 10.0) < 1e-6
