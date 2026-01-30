import numpy as np

def wrap_to_pi(theta: float) -> float:
    """将角度归一化到 [-pi, pi]"""
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)

def global_to_body(pos_global: np.ndarray, base_pos: np.ndarray, base_theta: float) -> np.ndarray:
    """
    将全局坐标点转换为相对于 base_pos 的车体坐标系坐标
    pos_global: [x, y]
    """
    rel_p = pos_global - base_pos
    c, s = np.cos(base_theta), np.sin(base_theta)
    rot_T = np.array([[c, s], [-s, c]])
    return rot_T @ rel_p