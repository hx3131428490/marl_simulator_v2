from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 类型别名：(x_index, y_index)
Cell = Tuple[int, int]


@dataclass
class OccupancyGrid:
    """
    占用栅格地图 (Occupancy Grid Map)
    负责处理连续世界坐标(米)与离散网格坐标(索引)之间的转换及存储。
    """
    width_cells: int  # 地图宽度 (网格数)
    height_cells: int  # 地图高度 (网格数)
    resolution: float  # 分辨率 (米/网格)
    origin_world: Tuple[float, float] = (0.0, 0.0)  # 地图左下角在世界坐标系中的位置

    def __post_init__(self) -> None:
        if self.width_cells <= 0 or self.height_cells <= 0:
            raise ValueError("width_cells and height_cells must be positive")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")

        # 初始化网格数据: 0=空闲, 1=占用 (障碍物)
        # 注意 numpy shape 是 (height, width) -> (y, x)
        self.occ: np.ndarray = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)

    @property
    def width_m(self) -> float:
        """地图物理宽度 (米)"""
        return self.width_cells * self.resolution

    @property
    def height_m(self) -> float:
        """地图物理高度 (米)"""
        return self.height_cells * self.resolution

    def in_bounds_cell(self, cell: Cell) -> bool:
        """检查网格索引是否在地图范围内"""
        ix, iy = cell
        return 0 <= ix < self.width_cells and 0 <= iy < self.height_cells

    def in_bounds_world(self, x: float, y: float) -> bool:
        """检查物理坐标是否在地图范围内"""
        ox, oy = self.origin_world
        return (ox <= x < ox + self.width_m) and (oy <= y < oy + self.height_m)

    def world_to_cell(self, x: float, y: float, *, strict: bool = True) -> Optional[Cell]:
        """
        核心逻辑：将物理坐标(米)转换为网格索引。
        原理: index = floor((pos - origin) / resolution)

        Args:
            strict: 若为True，坐标越界时抛出异常；若为False，越界时返回 None。
        """
        if not np.isfinite([x, y]).all():
            raise ValueError("world_to_cell received non-finite world coordinates")

        if not self.in_bounds_world(x, y):
            if strict:
                raise ValueError(f"world_to_cell out of bounds: x={x}, y={y}")
            return None

        ox, oy = self.origin_world
        ix = int(np.floor((x - ox) / self.resolution))
        iy = int(np.floor((y - oy) / self.resolution))

        # 双重保险：防止浮点数精度误差导致的越界
        if not (0 <= ix < self.width_cells and 0 <= iy < self.height_cells):
            if strict:
                raise ValueError(f"world_to_cell produced out-of-bounds cell: {(ix, iy)}")
            return None

        return ix, iy

    def cell_to_world(self, ix: int, iy: int, *, center: bool = True, strict: bool = True) -> Optional[
        Tuple[float, float]]:
        """
        核心逻辑：将网格索引转换为物理坐标(米)。

        Args:
            center: 若为True，返回该网格中心的坐标 (x+0.5*res)；
                    若为False，返回该网格左下角的坐标。
        """
        if not self.in_bounds_cell((ix, iy)):
            if strict:
                raise ValueError(f"cell_to_world out of bounds: {(ix, iy)}")
            return None

        ox, oy = self.origin_world
        base_x = ox + ix * self.resolution
        base_y = oy + iy * self.resolution

        # 如果需要中心点，加半个分辨率的偏移量
        if center:
            base_x += 0.5 * self.resolution
            base_y += 0.5 * self.resolution
        return float(base_x), float(base_y)

    def set_occupied_cell(self, ix: int, iy: int, value: bool = True) -> None:
        """设置某个格子为占用(True)或空闲(False)"""
        if not self.in_bounds_cell((ix, iy)):
            raise ValueError(f"set_occupied_cell out of bounds: {(ix, iy)}")
        # 注意: numpy 索引顺序是 [y, x]
        self.occ[iy, ix] = 1 if value else 0

    def is_occupied_cell(self, ix: int, iy: int, *, out_of_bounds_as_occupied: bool = True) -> bool:
        """
        检查某个格子是否被占用。

        Args:
            out_of_bounds_as_occupied: 关键参数。如果查询越界，是否视为障碍物？
                                       通常设为 True，相当于地图周围有一圈空气墙。
        """
        if not self.in_bounds_cell((ix, iy)):
            return bool(out_of_bounds_as_occupied)
        return bool(self.occ[iy, ix] != 0)

    def is_occupied_world(self, x: float, y: float, *, out_of_bounds_as_occupied: bool = True) -> bool:
        """直接检查物理坐标点是否被占用"""
        cell = self.world_to_cell(x, y, strict=False)
        if cell is None:
            return bool(out_of_bounds_as_occupied)
        ix, iy = cell
        return self.is_occupied_cell(ix, iy, out_of_bounds_as_occupied=out_of_bounds_as_occupied)