import numpy as np
from gym import spaces


class OnPolicyAdapter:
    """
    将 CoopFollowEnv 适配为 on-policy MAPPO 所用的 MPE 风格接口：
      reset() -> obs                  形状 (n_agents, obs_dim)
      step(a) -> obs, rew, done, info  rew 形状 (n_agents, 1), done 形状 (n_agents,)
    注意：share_obs 由 runner 基于 obs 在内部构造，不要在这里返回 share_obs。
    """

    def __init__(self, coop_env, Nv: int = 11, Nw: int = 11):
        self.env = coop_env
        self.num_agents = 2

        self.obs_dim = 8

        # 离散化分辨率
        self.Nv = int(Nv)
        self.Nw = int(Nw)
        assert self.Nv >= 2 and self.Nw >= 2
        self.K = self.Nv * self.Nw

        # 动作空间改为 Discrete(K)
        self.action_space = [
            spaces.Discrete(self.K)
            for _ in range(self.num_agents)
        ]

        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim * self.num_agents,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

    def render(self, mode="human"):
        # 这里的 self.env 是 CoopFollowEnv
        return self.env.render(mode)

    @staticmethod
    def _stack_obs_from_dict(obs_dict):
        obs0 = np.asarray(obs_dict[0], dtype=np.float32)
        obs1 = np.asarray(obs_dict[1], dtype=np.float32)
        obs = np.stack([obs0, obs1], axis=0)

        if obs.shape != (2, obs0.shape[-1]):
            raise ValueError(f"Bad obs shape: {obs.shape}, obs0={obs0.shape}, obs1={obs1.shape}")

        if not np.isfinite(obs).all():
            bad = np.argwhere(~np.isfinite(obs))
            bad_vals = obs[~np.isfinite(obs)]
            raise ValueError(
                f"Observation contains NaN/Inf. bad_indices={bad.tolist()} bad_vals={bad_vals.tolist()} obs={obs}"
            )
        return obs

    @staticmethod
    def _to_global_done(done_raw):
        if isinstance(done_raw, dict):
            if "__all__" in done_raw:
                return bool(done_raw["__all__"])
            return bool(any(bool(v) for v in done_raw.values()))
        if np.isscalar(done_raw):
            return bool(done_raw)
        arr = np.asarray(done_raw)
        return bool(np.any(arr))

    def reset(self, seed=None):
        obs_dict = self.env.reset(seed=seed)
        obs = self._stack_obs_from_dict(obs_dict)
        return obs

    def _decode_action_ids(self, actions) -> np.ndarray:
        """
        兼容三类输入：
        1) (n_agents,) 或 (n_agents,1) 的整数 id
        2) (n_agents, K) 的 one-hot 或近似 one-hot
        返回 shape (n_agents,) 的 int64 ids
        """
        a = np.asarray(actions)

        # one-hot
        if a.ndim == 2 and a.shape[0] == self.num_agents and a.shape[1] == self.K:
            ids = np.argmax(a, axis=1).astype(np.int64)
            return ids

        # (n_agents, 1)
        if a.ndim == 2 and a.shape[0] == self.num_agents and a.shape[1] == 1:
            ids = a.squeeze(1).astype(np.int64)
            return ids

        # (n_agents,)
        if a.ndim == 1 and a.shape[0] == self.num_agents:
            ids = a.astype(np.int64)
            return ids

        raise ValueError(f"Bad discrete actions shape: {a.shape}, expect (2,), (2,1), or (2,{self.K})")

    def _id_to_vw(self, a_id: int, v_max: float, w_max: float):
        a_id = int(np.clip(a_id, 0, self.K - 1))
        iv = a_id // self.Nw
        iw = a_id % self.Nw

        v_norm = -1.0 + 2.0 * (iv / (self.Nv - 1))
        w_norm = -1.0 + 2.0 * (iw / (self.Nw - 1))

        v = float(v_norm) * float(v_max)
        w = float(w_norm) * float(w_max)
        return v, w

    def step(self, actions):
        ids = self._decode_action_ids(actions)

        cfg = self.env.env.cfg
        v_max = float(cfg.agent.v_max)
        w_max = float(cfg.agent.w_max)

        v0, w0 = self._id_to_vw(ids[0], v_max, w_max)
        v1, w1 = self._id_to_vw(ids[1], v_max, w_max)

        follower_actions = {
            0: np.array([v0, w0], dtype=np.float32),
            1: np.array([v1, w1], dtype=np.float32),
        }

        obs_dict, reward, done_raw, info_raw = self.env.step(follower_actions)

        obs = self._stack_obs_from_dict(obs_dict)

        r = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        rewards = np.array([[r], [r]], dtype=np.float32)

        global_done = self._to_global_done(done_raw)
        dones = np.array([global_done, global_done], dtype=bool)

        info = {"n": [info_raw, info_raw]}

        return obs, rewards, dones, info
