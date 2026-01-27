from __future__ import annotations

from marl_sim.config.schema import SimConfig


def default_config() -> SimConfig:
    """
    创建一个包含默认参数的仿真配置对象。

    用途:
        用于快速启动测试或作为自定义配置的基准。
        此函数会自动调用 validate() 确保默认参数组合的合法性。

    Returns:
        SimConfig: 一个经过校验的、合法的配置实例。
    """
    cfg = SimConfig()  # 使用 schema 定义的默认值初始化
    cfg.validate()  # 执行完整性检查 (防御性编程)
    return cfg
