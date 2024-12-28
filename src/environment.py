import numpy as np
import matplotlib.pyplot as plt

class SymmetryEnv:
    # Define a small discrete set of actions
    # (2 mirror actions + 2 draw actions as an example)
    ACTIONS = [
        {"type": "draw", "pos": (10, 10)},
        {"type": "draw", "pos": (15, 15)},
        {"type": "mirror", "axis": 0},
        {"type": "mirror", "axis": 1},
    ]

    def __init__(self, size=32):
        self.size = size
        self.canvas = np.zeros((self.size, self.size), dtype=int)

    def reset(self):
        self.canvas = np.zeros((self.size, self.size), dtype=int)
        return self.canvas

    def step(self, action):
        """Expects action to be a dict with 'type' and additional keys."""
        if action["type"] == "draw":
            x, y = action["pos"]
            if 0 <= x < self.size and 0 <= y < self.size:
                self.canvas[x, y] = 1
        elif action["type"] == "mirror":
            # Flip along the specified axis
            self.canvas = np.flip(self.canvas, axis=action["axis"])

        reward = self.compute_reward()
        done = False  # Or define your own end condition
        return self.canvas, reward, done

    def compute_reward(self):
        """Compute how symmetric the canvas is left-to-right."""
        left = self.canvas[:, : self.size // 2]
        right = np.flip(self.canvas[:, self.size // 2 :], axis=1)
        symmetry_score = np.sum(left == right) / self.canvas.size
        return symmetry_score

    def render(self, title="Canvas State", filename=None):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.canvas, cmap="Greys", interpolation="nearest")
        plt.title(title)
        plt.axis("off")
        if filename:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
