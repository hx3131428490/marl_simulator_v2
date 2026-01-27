import numpy as np

from marl_sim.config.defaults import default_config
from marl_sim.env.multi_car_env import MultiCarEnv


def test_env_reset_and_step_api() -> None:
    cfg = default_config()
    cfg.render.enabled = False
    cfg.agent.n_agents = 3
    cfg.max_steps = 50
    cfg.obstacles.enabled = True
    cfg.obstacles.n_rect = 6

    env = MultiCarEnv(cfg)
    obs = env.reset(seed=0)

    assert isinstance(obs, dict)
    assert set(obs.keys()) == {0, 1, 2}
    assert obs[0].shape == (cfg.agent.n_agents * 5,)

    actions = {i: np.array([0.2, 0.1], dtype=np.float64) for i in range(cfg.agent.n_agents)}
    out = env.step(actions)

    assert isinstance(out.obs, dict)
    assert isinstance(out.reward, dict)
    assert isinstance(out.done, dict)
    assert isinstance(out.info, dict)

    assert set(out.obs.keys()) == {0, 1, 2}
    assert set(out.reward.keys()) == {0, 1, 2}
    assert set(out.done.keys()) == {0, 1, 2}

    assert out.obs[1].shape == (cfg.agent.n_agents * 5,)
    assert all(np.isfinite(out.obs[i]).all() for i in range(cfg.agent.n_agents))
