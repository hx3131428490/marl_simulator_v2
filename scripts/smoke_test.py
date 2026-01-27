from __future__ import annotations

import numpy as np

from marl_sim.config import default_config
from marl_sim.utils import seed_everything, get_logger


def main() -> None:
    logger = get_logger("smoke_test")

    cfg = default_config()
    logger.info(f"Config loaded. dt={cfg.dt}, max_steps={cfg.max_steps}, n_agents={cfg.agent.n_agents}")

    rng = seed_everything(cfg.seed)
    a = rng.standard_normal(5)
    b = rng.standard_normal(5)

    logger.info(f"RNG sample a={np.array2string(a, precision=3)}")
    logger.info(f"RNG sample b={np.array2string(b, precision=3)}")
    logger.info("Smoke test finished without exceptions.")


if __name__ == "__main__":
    main()
