from marl_sim.config.defaults import default_config
from marl_sim.config.schema import SimConfig, MapConfig, AgentConfig, RenderConfig, CommConfig

# 定义模块导出的“白名单”
# 当用户执行 "from marl_sim.config import *" 时，只会导入以下内容。
# 这也允许用户直接通过 "from marl_sim.config import SimConfig" 访问类，
# 而不需要写全路径 "from marl_sim.config.schema import SimConfig"。
__all__ = [
    "default_config",
    "SimConfig",
    "MapConfig",
    "AgentConfig",
    "RenderConfig",
    "CommConfig",
]