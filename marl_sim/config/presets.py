from __future__ import annotations

from marl_sim.config.schema import SimConfig, MapConfig, AgentConfig, RenderConfig, CommConfig


def small_debug(seed: int = 0) -> SimConfig:
    """
    预设配置：小型调试模式。
    特点：地图更小、步数更少、像素更大，方便肉眼观察和快速迭代测试逻辑。
    """
    cfg = SimConfig(
        seed=seed,
        dt=0.05,
        max_steps=200,      # 缩短单局时长，快速结束
        map=MapConfig(width_cells=50, height_cells=40, resolution=0.1), # 缩小地图
        agent=AgentConfig(n_agents=3, radius=0.25, v_max=1.0, w_max=1.5), # 减少智能体
        render=RenderConfig(enabled=True, fps=60, cell_px=14), # 放大显示(cell_px)以便观察细节
        comm=CommConfig(enabled=True, range_m=5.0, require_los=True),
    )
    cfg.validate()
    return cfg


def headless_fast(seed: int = 0) -> SimConfig:
    """
    预设配置：无头急速模式 (通常用于训练)。
    特点：关闭图形渲染，以获得最快的仿真速度，适合在服务器后台运行大规模训练。
    """
    cfg = SimConfig(seed=seed) # 使用默认参数
    cfg.render.enabled = False # 显式关闭渲染以提升性能
    cfg.validate()
    return cfg