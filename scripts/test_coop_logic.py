import numpy as np
from marl_sim.config.defaults import default_config
from marl_sim.envs.coop_follow_env import CoopFollowEnv
from marl_sim.envs.onpolicy_adapter import OnPolicyAdapter


def test_run():
    cfg = default_config()
    cfg.agent.n_agents = 3
    cfg.obstacles.enabled = False  # 确保第四步配置已生效

    # 初始化环境
    base_env = CoopFollowEnv(cfg)
    env = OnPolicyAdapter(base_env)

    obs, share_obs, _ = env.reset()
    print(f"初始观测形状: {obs.shape}, 全局观测形状: {share_obs.shape}")

    for i in range(10):
        # 模拟随机动作
        random_actions = np.random.uniform(-1, 1, (2, 2))
        next_obs, next_share_obs, rewards, dones, infos, _ = env.step(random_actions)

        print(f"步数: {i}, 奖励: {rewards[0][0]:.4f}, 碰撞状态: {infos[0]['collided']}")

        if any(dones):
            print("环境已重置")
            env.reset()


if __name__ == "__main__":
    test_run()