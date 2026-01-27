from __future__ import annotations

import numpy as np

# 导入之前定义的所有模块
from marl_sim.config.defaults import default_config
from marl_sim.env.multi_car_env import MultiCarEnv
# 假设 PygameRenderer 位于此路径 (它是唯一还未展示代码的模块)
from marl_sim.render.pygame_renderer import PygameRenderer


def wrap_to_pi(theta: float) -> float:
    """角度归一化辅助函数"""
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def goal_seeking_actions(env: MultiCarEnv) -> dict:
    """
    基于规则的导航策略 (Heuristic Policy)。
    使用简单的 P控制 (Proportional Control) 让智能体驶向目标。
    用于测试环境物理特性是否正常。
    """
    actions = {}
    v_max = float(env.cfg.agent.v_max)
    w_max = float(env.cfg.agent.w_max)

    k_w = 2.0  # 转向灵敏度系数

    for i in range(env.n_agents):
        s = env.states[i]
        gx, gy = env.goals[i]

        # 1. 计算指向目标的向量
        dx = float(gx - s.x)
        dy = float(gy - s.y)

        # 2. 计算目标朝向角 (Arctan2)
        desired = float(np.arctan2(dy, dx))
        # 计算当前朝向与目标朝向的偏差
        err = wrap_to_pi(desired - float(s.theta))

        # 3. 计算角速度 (w): 偏差越大，转得越快
        w = float(np.clip(k_w * err, -w_max, w_max))

        # 4. 计算线速度 (v):
        # 智能调速策略：如果你正对着目标，全速前进；
        # 如果你背对着目标 (err接近pi)，停车原地旋转。
        v_scale = max(0.0, 1.0 - abs(err) / np.pi)
        v = float(v_max * v_scale)

        actions[i] = np.array([v, w], dtype=np.float64)

    return actions


def main() -> None:
    """主程序入口：启动仿真并渲染"""

    # 1. 配置环境
    cfg = default_config()
    cfg.render.enabled = True
    cfg.render.fps = 60
    cfg.render.cell_px = 12  # 每个网格占12像素

    cfg.agent.n_agents = 3  # 放置5个智能体
    cfg.max_steps = 2000  # 增加每局时长

    # 启用障碍物生成
    cfg.obstacles.enabled = True
    cfg.obstacles.n_rect = 4
    cfg.obstacles.w_range = (3, 8)
    cfg.obstacles.h_range = (3, 8)
    cfg.obstacles.margin_cells = 1

    # 2. 初始化环境
    env = MultiCarEnv(cfg)
    _ = env.reset(seed=0)

    # 3. 初始化渲染器 (负责画图)
    renderer = PygameRenderer(
        grid=env.grid,
        cell_px=int(cfg.render.cell_px),
        fps=int(cfg.render.fps),
        caption="marl_simulator_v1 demo",
    )

    running = True
    last_collided = [False for _ in range(env.n_agents)]

    print("Simulation started. Close the window to exit.")

    # 4. 仿真主循环 (Game Loop)
    while running:
        # 处理窗口事件 (如点击关闭按钮)
        running = renderer.pump_events()
        if not running:
            break

        # === 核心逻辑 ===
        # A. 计算动作 (这里用的是规则脚本，未来可以用神经网络)
        actions = goal_seeking_actions(env)

        # B. 环境步进 (物理积分、碰撞检测)
        out = env.step(actions)

        # C. 更新状态记录
        last_collided = [bool(out.info["collided"][i]) for i in range(env.n_agents)]

        # D. 准备渲染信息
        info_lines = []
        n_reached = sum(1 for i in range(env.n_agents) if bool(out.info["reached"][i]))
        info_lines.append(f"reached={n_reached}/{env.n_agents}")

        # E. 绘制画面
        renderer.draw(
            states=env.states,
            goals=env.goals,
            step_count=int(out.info["step_count"]),
            collided=last_collided,
            edges=list(out.info["edges"]),
            info_lines=info_lines,
        )

        # F. 自动重置判定
        if all(out.done.values()):
            print("Episode finished. Resetting...")
            _ = env.reset(seed=0)  # 这里种子写死0是为了演示，实际训练可以去掉
            # 重要：重置后地图变了，渲染器需要重新生成背景图
            renderer.rebuild_static_map()

    # 5. 退出清理
    renderer.close()
    env.close()


if __name__ == "__main__":
    main()