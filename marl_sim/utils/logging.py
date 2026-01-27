from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str = "marl_sim", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    获取或创建一个配置好的日志记录器。

    功能特点:
    1. 单例模式: 多次调用不会导致日志重复打印 (通过检查 handlers 实现)。
    2. 双重输出: 支持同时输出到控制台(Console)和文件(File)。
    3. 自动建路: 如果日志文件的目录不存在，会自动创建。
    4. 独立性: 关闭 propagate，防止日志向上传播导致重复。

    Args:
        name: 日志记录器的名称 (通常用于区分不同模块)。
        log_file: 日志文件的保存路径。如果为 None，则只输出到控制台。
        level: 日志级别 (如 logging.INFO, logging.DEBUG)。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 禁止向根记录器传播，防止重复打印

    # === 关键检查 ===
    # 如果该 Logger 已经有处理器(Handlers)，说明已经被初始化过。
    # 直接返回，防止重复添加 Handler 导致一条日志打印多次。
    if logger.handlers:
        return logger

    # 定义统一的日志格式: [时间] | [级别] | [模块名] | [消息]
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # === 1. 控制台处理器 (输出到屏幕) ===
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # === 2. 文件处理器 (输出到文件) ===
    if log_file is not None:
        # 自动创建父级目录，防止因文件夹不存在而报错
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger