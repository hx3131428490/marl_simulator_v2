import numpy as np

from marl_sim.config.defaults import default_config
from marl_sim.env.multi_car_env import MultiCarEnv
from marl_sim.agent.controllers import GoalSeekingController
from marl_sim.utils.trajectory import TrajectoryRecorder


def test_trajectory_shapes(tmp_path) -> None:
    cfg = default_config()
    cfg.render.enabled = False
    cfg.agent.n_agents = 3
    cfg.max_steps = 30
    cfg.obstacles.enabled = True
    cfg.obstacles.n_rect = 6

    env = MultiCarEnv(cfg)
    _ = env.reset(seed=0)

    ctrl = GoalSeekingController(k_w=2.0)
    rec = TrajectoryRecorder(n_agents=cfg.agent.n_agents)
    rec.reset_episode(env)

    for _ in range(10):
        actions = ctrl.act(env)
        out = env.step(actions)
        rec.record_step(env, actions, out)

    data = rec.to_npz_dict()
    assert data["states"].shape[1:] == (cfg.agent.n_agents, 3)
    assert data["goals"].shape == (cfg.agent.n_agents, 2)
    assert data["actions"].shape[1:] == (cfg.agent.n_agents, 2)

    p = tmp_path / "traj.npz"
    rec.save_npz(p)
    loaded = np.load(p, allow_pickle=True)
    assert loaded["states"].ndim == 3
    assert loaded["goals"].shape == (cfg.agent.n_agents, 2)
