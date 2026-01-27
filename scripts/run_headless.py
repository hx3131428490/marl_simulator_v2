from __future__ import annotations

import numpy as np

from marl_sim.config.defaults import default_config
from marl_sim.env.multi_car_env import MultiCarEnv


def main() -> None:
    cfg = default_config()
    cfg.render.enabled = False
    cfg.agent.n_agents = 4
    cfg.max_steps = 200
    cfg.obstacles.enabled = True
    cfg.obstacles.n_rect = 10

    env = MultiCarEnv(cfg)
    obs = env.reset(seed=0)

    ep_ret = {i: 0.0 for i in range(cfg.agent.n_agents)}

    for t in range(cfg.max_steps):
        actions = {}
        for i in range(cfg.agent.n_agents):
            v = float(np.random.uniform(-cfg.agent.v_max, cfg.agent.v_max))
            w = float(np.random.uniform(-cfg.agent.w_max, cfg.agent.w_max))
            actions[i] = np.array([v, w], dtype=np.float64)

        out = env.step(actions)
        for i in range(cfg.agent.n_agents):
            ep_ret[i] += float(out.reward[i])

        if all(out.done.values()):
            break

    print("Episode return per agent:", ep_ret)


if __name__ == "__main__":
    main()
