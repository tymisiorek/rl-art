# src/environment.py
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import itertools

class ArtEnv:
    """
    Environment rewarding coverage and coherent shapes, 
    rather than symmetry.
    """
    def __init__(self, size=16):
        self.size = size
        self.canvas = np.zeros((self.size, self.size), dtype=float)  # Ensure float type for normalization
        self.generate_actions()

    def generate_actions(self):
        """
        Generate a larger set of actions for a 16x16 canvas.
        """
        step = 2  # Decreased step to 2 for more actions
        draw_positions = [
            (x, y) for x, y in itertools.product(range(0, self.size, step), range(0, self.size, step))
        ]
        draw_actions = [{"type": "draw", "pos": pos} for pos in draw_positions]
        mirror_actions = [
            {"type": "mirror", "axis": 0},
            {"type": "mirror", "axis": 1},
            {"type": "rotate", "angle": 90},  # New rotate action
        ]
        draw_line_actions = [
            {"type": "draw_line", "start": (0, 0), "end": (self.size-1, self.size-1)},
            {"type": "draw_line", "start": (0, self.size-1), "end": (self.size-1, 0)},
        ]
        self.ACTIONS = draw_actions + mirror_actions + draw_line_actions  # Expanded action set

    def reset(self):
        """Reset the canvas to all zeros."""
        self.canvas = np.zeros((self.size, self.size), dtype=float)
        return self.canvas.copy()  # Return a copy to ensure contiguous array

    def step(self, action):
        """
        Execute the given action on the canvas.
        Return: (next_state, reward, done)
        """
        reward = 0.0

        # Immediate reward for drawing a new pixel
        if action["type"] == "draw":
            x, y = action["pos"]
            if 0 <= x < self.size and 0 <= y < self.size:
                if self.canvas[x, y] == 0:
                    reward += 1.0  # Reward for new pixel
                else:
                    reward -= 0.5  # Penalty for overwriting
                self.canvas[x, y] = 1

        elif action["type"] == "mirror":
            # Flip along the specified axis (0 => vertical flip, 1 => horizontal flip)
            self.canvas = np.flip(self.canvas, axis=action["axis"])

        elif action["type"] == "rotate":
            # Rotate the canvas by the specified angle
            angle = action["angle"]
            k = angle // 90  # Ensure angle is a multiple of 90
            self.canvas = np.rot90(self.canvas, k=k)
            # No immediate reward for rotation

        elif action["type"] == "draw_line":
            # Accumulate reward from drawing the line
            reward += self._draw_line(action["start"][0], action["start"][1],
                                      action["end"][0], action["end"][1])

        # Compute rewards
        coverage_reward = np.sum(self.canvas) / (self.size * self.size)
        shape_reward = self._compute_shape_measure()

        # Weighted sum: higher weight on coverage
        reward += 0.8 * coverage_reward + 0.2 * shape_reward

        # Define terminal condition
        done = coverage_reward >= 0.95  # End episode if 95% coverage

        return self.canvas.copy(), reward, done  # Return a copy to ensure contiguous array

    def _draw_line(self, x1, y1, x2, y2):
        """
        Simple line drawing. Returns the accumulated reward from drawing new pixels.
        """
        reward = 0.0
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            # Single point
            if 0 <= x1 < self.size and 0 <= y1 < self.size:
                if self.canvas[x1, y1] == 0:
                    reward += 1.0  # Increased
                self.canvas[x1, y1] = 1
            return reward

        x_inc = dx / steps
        y_inc = dy / steps
        x_cur, y_cur = x1, y1
        for _ in range(int(steps) + 1):
            ix, iy = int(round(x_cur)), int(round(y_cur))
            if 0 <= ix < self.size and 0 <= iy < self.size:
                if self.canvas[ix, iy] == 0:
                    reward += 1.0  # Increased
                self.canvas[ix, iy] = 1
            x_cur += x_inc
            y_cur += y_inc

        return reward

    def _compute_shape_measure(self):
        """
        Finds connected components, measures how 'solid' each component is 
        (component area / bounding box area), then averages across components.
        Range: 0.0 (nothing or very fragmented) to 1.0 (every shape is 'solid').
        """
        labels = measure.label(self.canvas, background=0, connectivity=1)
        props = measure.regionprops(labels)

        if not props:
            return 0.0  # No shapes at all

        shape_scores = []
        for region in props:
            region_area = region.area
            minr, minc, maxr, maxc = region.bbox
            bbox_area = (maxr - minr) * (maxc - minc)
            if bbox_area > 0:
                shape_scores.append(region_area / bbox_area)
            else:
                shape_scores.append(0)

        # Average shape compactness across all components
        return np.mean(shape_scores)

    def render(self, title="Canvas State", filename=None):
        """
        Render the current canvas. Ensures consistent image size by fixing figsize and dpi.
        """
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # Increased figure size for larger canvas
        ax.imshow(self.canvas, cmap="Greys", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

        if filename:
            fig.savefig(filename, bbox_inches="tight", pad_inches=0)
        else:
            plt.show()
        plt.close(fig)
