from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from marl_sim.config.defaults import default_config
from marl_sim.env.multi_car_env import MultiCarEnv
from marl_sim.agent.controllers import RandomController, GoalSeekingController
from marl_sim.utils import TrajectoryRecorder


def main() -> None:
    cfg = default_config()
    cfg.render.enabled = False
    cfg.agent.n_agents = 5
    cfg.max_steps = 400

    cfg.obstacles.enabled = True
    cfg.obstacles.n_rect = 14
    cfg.obstacles.w_range = (3, 8)
    cfg.obstacles.h_range = (3, 8)
    cfg.obstacles.margin_cells = 1

    env = MultiCarEnv(cfg)
    _ = env.reset(seed=0)

    rng = np.random.default_rng(0)
    controller = GoalSeekingController(k_w=2.0)
    # controller = RandomController(rng=rng, v_max=cfg.agent.v_max, w_max=cfg.agent.w_max)

    rec = TrajectoryRecorder(n_agents=cfg.agent.n_agents)
    rec.reset_episode(env)

    for _t in range(cfg.max_steps):
        actions = controller.act(env)
        out = env.step(actions)
        rec.record_step(env, actions, out)
        if all(out.done.values()):
            break

    # === 修改部分开始 ===
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 获取当前脚本所在的目录 (即: .../your_project/scripts)
    script_dir = Path(__file__).resolve().parent

    # 2. 获取项目根目录 (即: .../your_project)
    #    通过 .parent 回退一级，找到 scripts 的父目录
    project_root = script_dir.parent

    # 3. 拼接 outputs 路径 (即: .../your_project/outputs)
    out_dir = project_root / "outputs"

    # 4. 组合最终文件路径
    path = out_dir / f"traj_{ts}.npz"
    # === 修改部分结束 ===

    # save_npz 会自动创建父文件夹(如果不存在)
    rec.save_npz(path)

    print("Saved:", str(path))


if __name__ == "__main__":
    main()