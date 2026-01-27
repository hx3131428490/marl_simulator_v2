from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pygame

from marl_sim.agent.state import CarState
from marl_sim.world.grid_map import OccupancyGrid
from marl_sim.render import colors


@dataclass
class PygameRenderer:
    grid: OccupancyGrid
    cell_px: int = 12
    fps: int = 60
    caption: str = "marl_simulator_v1"

    def __post_init__(self) -> None:
        if self.cell_px <= 0:
            raise ValueError("cell_px must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")

        pygame.init()
        pygame.display.set_caption(self.caption)

        self.w_px = int(self.grid.width_cells * self.cell_px)
        self.h_px = int(self.grid.height_cells * self.cell_px)

        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self._static_map_surface: Optional[pygame.Surface] = None
        self.rebuild_static_map()

    def rebuild_static_map(self) -> None:
        surface = pygame.Surface((self.w_px, self.h_px))
        surface.fill(colors.BG)

        for ix in range(self.grid.width_cells + 1):
            x = int(ix * self.cell_px)
            pygame.draw.line(surface, colors.GRID_LINE, (x, 0), (x, self.h_px), 1)
        for iy in range(self.grid.height_cells + 1):
            y = int(iy * self.cell_px)
            pygame.draw.line(surface, colors.GRID_LINE, (0, y), (self.w_px, y), 1)

        occ = self.grid.occ
        for iy in range(self.grid.height_cells):
            for ix in range(self.grid.width_cells):
                if occ[iy, ix] == 0:
                    continue
                x0 = int(ix * self.cell_px)
                y0 = int((self.grid.height_cells - 1 - iy) * self.cell_px)
                pygame.draw.rect(surface, colors.OBSTACLE, pygame.Rect(x0, y0, self.cell_px, self.cell_px))

        self._static_map_surface = surface

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        ox, oy = self.grid.origin_world
        u = (x - ox) / self.grid.resolution
        v = (y - oy) / self.grid.resolution

        sx = int(u * self.cell_px)
        sy = int((self.grid.height_cells - v) * self.cell_px)
        return sx, sy

    def draw(
            self,
            states: List[CarState],
            goals: List[Tuple[float, float]],
            *,
            step_count: int = 0,
            collided: Optional[List[bool]] = None,
            # [新增] 通信边列表
            edges: Optional[List[Tuple[int, int]]] = None,
            info_lines: Optional[List[str]] = None,
    ) -> None:
        if self._static_map_surface is None:
            self.rebuild_static_map()

        self.screen.blit(self._static_map_surface, (0, 0))

        if collided is None:
            collided = [False for _ in states]

        # [新增] 处理 edges 默认值
        if edges is None:
            edges = []

        # 1. 绘制目标点 (Goals)
        for i, (gx, gy) in enumerate(goals):
            sx, sy = self.world_to_screen(float(gx), float(gy))
            pygame.draw.circle(self.screen, colors.GOAL, (sx, sy), 5, 0)

        # [新增] 2. 绘制通信连线 (Communication Edges)
        # 建议在画车之前画线，这样线会被车压在下面，视觉上更自然
        for (i, j) in edges:
            si = states[i]
            sj = states[j]
            x0, y0 = self.world_to_screen(float(si.x), float(si.y))
            x1, y1 = self.world_to_screen(float(sj.x), float(sj.y))
            # 颜色需要在 colors.py 中定义 COMM_EDGE，或者直接写 (0, 255, 255) 青色
            # 假设您已在 colors.py 定义了 COMM_EDGE = (0, 200, 200)
            # 如果没有，可以直接用 RGB 元组
            line_color = getattr(colors, 'COMM_EDGE', (0, 200, 200))
            pygame.draw.line(self.screen, line_color, (x0, y0), (x1, y1), 2)

        # 3. 绘制智能体 (Agents)
        for i, s in enumerate(states):
            sx, sy = self.world_to_screen(float(s.x), float(s.y))

            col = colors.AGENT_ALT if bool(collided[i]) else colors.AGENT
            pygame.draw.circle(self.screen, col, (sx, sy), 7, 0)

            hx = float(s.x + 0.6 * np.cos(float(s.theta)) * self.grid.resolution * 3.0)
            hy = float(s.y + 0.6 * np.sin(float(s.theta)) * self.grid.resolution * 3.0)
            hx_s, hy_s = self.world_to_screen(hx, hy)
            pygame.draw.line(self.screen, colors.AGENT_HEADING, (sx, sy), (hx_s, hy_s), 2)

        # 4. 绘制文字信息 (HUD)
        lines = [f"step={int(step_count)}", f"agents={len(states)}"]
        if info_lines:
            lines.extend([str(x) for x in info_lines])

        y = 6
        for t in lines:
            surf = self.font.render(t, True, colors.TEXT)
            self.screen.blit(surf, (6, y))
            y += 18

        pygame.display.flip()
        self.clock.tick(self.fps)

    def pump_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def close(self) -> None:
        pygame.quit()