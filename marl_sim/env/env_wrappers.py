import numpy as np
from marl_sim.config.defaults import default_config
from marl_sim.env.coop_follow_env import CoopFollowEnv
from marl_sim.env.onpolicy_adapter import OnPolicyAdapter
from marl_sim.third_party.onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            cfg = default_config()
            # 根据 rank 设置不同的随机种子，保证采样多样性
            cfg.seed = all_args.seed + rank * 1000
            base_env = CoopFollowEnv(cfg)
            env = OnPolicyAdapter(base_env)
            return env
        return init_env

    # 根据配置的线程数选择并行方式
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])