import numpy as np
import matplotlib.pyplot as plt

# OPTIONAL: If you want to draw lines easily, you can use skimage (if installed).
# from skimage.draw import line  # or implement a simple line-drawing method below.

class SymmetryEnv:
    """
    Environment with a discrete set of actions:
        - 2 mirror actions (flip axis=0 or axis=1)
        - 2 single-point draw actions
        - 1 line-drawing action
        (Optionally remove "change_color" if not using color canvas.)
    """
    ACTIONS = [
        {"type": "draw", "pos": (10, 10)},
        {"type": "draw", "pos": (20, 20)},
        {"type": "mirror", "axis": 0},
        {"type": "mirror", "axis": 1},
        {"type": "draw_line", "start": (5, 5), "end": (15, 15)},
        # Remove or implement "change_color" if you are not handling color:
        # {"type": "change_color", "color": (255, 0, 0)},
    ]

    def __init__(self, size=32):
        self.size = size
        self.canvas = np.zeros((self.size, self.size), dtype=int)

    def reset(self):
        """Reset the canvas to all zeros."""
        self.canvas = np.zeros((self.size, self.size), dtype=int)
        return self.canvas

    def step(self, action):
        """
        Execute the given action on the canvas.
        Return: (next_state, reward, done)
        """
        reward = 0.0

        if action["type"] == "draw":
            x, y = action["pos"]
            if 0 <= x < self.size and 0 <= y < self.size:
                # Reward for drawing on an empty pixel
                if self.canvas[x, y] == 0:
                    reward += 0.1
                self.canvas[x, y] = 1

        elif action["type"] == "mirror":
            # Flip along the specified axis (0 => vertical flip, 1 => horizontal flip)
            self.canvas = np.flip(self.canvas, axis=action["axis"])

        elif action["type"] == "draw_line":
            x1, y1 = action["start"]
            x2, y2 = action["end"]
            # Simple naive line-drawing approach (only handles x1<=x2, y1<=y2):
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                # Just a single point
                if 0 <= x1 < self.size and 0 <= y1 < self.size:
                    if self.canvas[x1, y1] == 0:
                        reward += 0.1
                    self.canvas[x1, y1] = 1
            else:
                steps = max(abs(dx), abs(dy))
                # Avoid dividing by zero
                if steps == 0:
                    steps = 1
                x_inc = dx / steps
                y_inc = dy / steps
                x_cur, y_cur = x1, y1

                for _ in range(int(steps) + 1):
                    ix, iy = int(round(x_cur)), int(round(y_cur))
                    if 0 <= ix < self.size and 0 <= iy < self.size:
                        if self.canvas[ix, iy] == 0:
                            reward += 0.1
                        self.canvas[ix, iy] = 1
                    x_cur += x_inc
                    y_cur += y_inc

        # (Optional) handle change_color if you really have a color canvas
        # elif action["type"] == "change_color":
        #     # Not implemented in a simple 2D int canvas

        # Compute combined reward: coverage + symmetry
        coverage_reward = np.sum(self.canvas) / (self.size * self.size)
        symmetry_reward = self._compute_symmetry_score()
        # Weighted sum: encourage filling + symmetry
        reward += 0.5 * coverage_reward + 0.5 * symmetry_reward

        done = False  # You can define a stopping condition if you like
        return self.canvas, reward, done

    def _compute_symmetry_score(self):
        """
        Returns how symmetric the canvas is from left to right.
        Range: 0.0 (no symmetry) to 1.0 (perfectly symmetric).
        """
        left = self.canvas[:, : self.size // 2]
        right = np.flip(self.canvas[:, self.size // 2 :], axis=1)
        return np.sum(left == right) / self.canvas.size

    def render(self, title="Canvas State", filename=None):
        """
        Render the current canvas. For large canvases, increase figsize.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.canvas, cmap="Greys", interpolation="nearest")
        plt.title(title)
        plt.axis("off")
        if filename:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
