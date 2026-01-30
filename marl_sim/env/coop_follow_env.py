import numpy as np
from marl_sim.env.multi_car_env import MultiCarEnv
from marl_sim.utils.math2d import global_to_body, wrap_to_pi
from marl_sim.env.leader_controller import LeaderController


class CoopFollowEnv:
    def __init__(self, cfg):
        self.env = MultiCarEnv(cfg)
        cfg.agent.n_agents = 3
        self.leader_ctrl = LeaderController(v_max=cfg.agent.v_max, w_max=cfg.agent.w_max)

        # 两台从车在头车坐标系下的期望槽位
        self.slots = [
            np.array([-0.8, 0.6], dtype=np.float32),
            np.array([-0.8, -0.6], dtype=np.float32),
        ]

        # 未参与训练的额外智能体默认静止
        self._idle_action = np.zeros(2, dtype=np.float32)

        # 势函数差分奖励的缓存
        self._last_potential = 0.0

    def render(self, mode="human"):
        # 确保底层 MultiCarEnv 的配置开启了渲染
        self.env.cfg.render.enabled = True
        return self.env.render(mode)

    @staticmethod
    def _sanitize(x: np.ndarray, clip: float = 1e6) -> np.ndarray:
        return np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).astype(np.float32)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.leader_ctrl.reset()
        self._last_potential = float(self._potential())
        return self._get_coop_obs()

    def step(self, follower_actions):
        """
        follower_actions 只包含两台从车的动作，键为 0 和 1
        follower_actions[0] -> env agent 1
        follower_actions[1] -> env agent 2
        """
        a0 = np.asarray(self.leader_ctrl.get_action(), dtype=np.float32)

        # 先按 MultiCarEnv 的总智能体数补齐所有动作
        n_total = getattr(self.env, "n_agents", None)
        if n_total is None:
            n_total = len(getattr(self.env, "states", []))
        if n_total is None or n_total <= 0:
            n_total = 4  # 兜底

        all_actions = {i: self._idle_action.copy() for i in range(int(n_total))}

        # leader
        all_actions[0] = a0

        # 两台从车
        all_actions[1] = np.asarray(follower_actions[0], dtype=np.float32)
        all_actions[2] = np.asarray(follower_actions[1], dtype=np.float32)

        step_res = self.env.step(all_actions)

        reward = float(self._compute_coop_reward())
        obs = self._get_coop_obs()
        dones = step_res.done
        return obs, reward, dones, step_res.info

    def _potential(self) -> float:
        """
        以两台从车到其目标槽位的负距离和作为势函数
        """
        states = self.env.states
        leader = states[0]

        c = float(np.cos(leader.theta))
        s = float(np.sin(leader.theta))
        R = np.array([[c, -s], [s, c]], dtype=np.float32)

        pot = 0.0
        leader_xy = np.array([leader.x, leader.y], dtype=np.float32)

        for env_id, slot in zip([1, 2], self.slots):
            follower = states[env_id]
            desired = leader_xy + R @ slot
            err = np.array([follower.x, follower.y], dtype=np.float32) - desired
            dist = float(np.linalg.norm(err))
            pot += -dist

        pot = float(np.nan_to_num(pot, nan=0.0, posinf=0.0, neginf=0.0))
        return pot

    def _compute_coop_reward(self) -> float:
        """
        势函数差分奖励
        """
        pot = float(self._potential())
        r = pot - float(self._last_potential)
        self._last_potential = pot
        return float(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))

    def _get_coop_obs(self):
        """
        每台从车 8 维观测：
        theta, cos(theta), sin(theta),
        leader_rel_x, leader_rel_y,
        rel_yaw,
        slot_x, slot_y
        """
        states = self.env.states
        obs = {}

        for i in [1, 2]:
            me = states[i]
            leader = states[0]

            p_l_rel = global_to_body(
                np.array([leader.x, leader.y], dtype=np.float32),
                np.array([me.x, me.y], dtype=np.float32),
                float(me.theta),
            )
            p_l_rel = self._sanitize(np.asarray(p_l_rel, dtype=np.float32))

            my_slot = self.slots[i - 1]

            feat = np.array([
                float(me.theta),
                float(np.cos(me.theta)),
                float(np.sin(me.theta)),
                float(p_l_rel[0]),
                float(p_l_rel[1]),
                float(wrap_to_pi(leader.theta - me.theta)),
                float(my_slot[0]),
                float(my_slot[1]),
            ], dtype=np.float32)

            obs[i - 1] = self._sanitize(feat)

        return obs
