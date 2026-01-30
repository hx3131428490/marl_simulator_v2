import sys
import os
import argparse
import numpy as np
import torch
import pathlib

# 1. 自动定位路径：确保能够找到 marl_sim 文件夹和 third_party 下的 on-policy
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
third_party_path = os.path.join(project_root, "marl_sim", "third_party")

if third_party_path not in sys.path:
    sys.path.insert(0, third_party_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 导入 onpolicy 提供的配置和运行器
from marl_sim.third_party.onpolicy.config import get_config
from marl_sim.third_party.onpolicy.runner.shared.mpe_runner import MPERunner as Runner
from marl_sim.env.env_wrappers import make_train_env
from marl_sim.env.leader_controller import LeaderController


def parse_args(args, parser):
    """解析并设置任务参数"""
    parser.add_argument('--scenario_name', type=str, default="coop_follow")
    parser.add_argument('--num_agents', type=int, default=2)

    # 允许在命令行未指定时使用默认值，避免 parse_known_args 丢失关键项
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # 设置计算设备
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # --- 修复逻辑：计算并创建路径对象 (解决 TypeError: unsupported operand type(s) for /) ---
    # 必须确保 run_dir 是 pathlib.Path 对象，而不是字符串
    model_dir = pathlib.Path(project_root) / "results"
    run_dir = model_dir / all_args.env_name / all_args.algorithm_name / all_args.experiment_name

    if not run_dir.exists():
        # 创建目录时需要转换为字符串，但之后存入 config 必须保持 Path 对象
        os.makedirs(str(run_dir), exist_ok=True)
    # ----------------------------------------------------------------------------------

    # 1. 创建并行环境
    envs = make_train_env(all_args)
    eval_envs = make_train_env(all_args) if all_args.use_eval else None

    # 2. 构造 Runner 字典
    # 注意：run_dir 传给 Runner 时必须是 Path 类型，因为 Runner 内部会执行 run_dir / 'logs'
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "num_agents": all_args.num_agents,
        "run_dir": run_dir,  # 修复点：直接传入 Path 对象
    }

    # 3. 实例化运行器并开始训练
    runner = Runner(config)
    runner.run()

    # 4. 训练结束清理
    envs.close()
    if all_args.use_eval:
        eval_envs.close()


if __name__ == "__main__":
    # 执行前提醒：请确保已执行 pip install "numpy<2.0" 以兼容 Gym
    main(sys.argv[1:])