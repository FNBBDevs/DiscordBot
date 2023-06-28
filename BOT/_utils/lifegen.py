import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LifeGen:
    def __init__(self, decay=True):
        self.decay = decay
        self.color_maps = plt.colormaps()
        self.color_maps_lower = {
            cmap.lower(): i for i, cmap in enumerate(self.color_maps)
        }
        self.color_maps_upper = {
            cmap.upper(): i for i, cmap in enumerate(self.color_maps)
        }
        self.interps = [
            "antialiased",
            "none",
            "nearest",
            "bilinear",
            "bicubic",
            "spline16",
            "spline36",
            "hanning",
            "hamming",
            "hermite",
            "kaiser",
            "quadric",
            "catrom",
            "gaussian",
            "bessel",
            "mitchell",
            "sinc",
            "lanczos",
            "blackman",
        ]
        self.directions = [
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ]
        self.phases = (
            {**{i: i - 10 for i in range(200, 0, -10)}, **{0: 0}} if decay else {0: 0}
        )
        self.phase_set = 190 if decay else 0

    def randomGrid(self, N):
        """this function generates an NxN list of random values"""
        return np.random.choice(
            [200, 0],
            N * N,
        ).reshape(N, N)

    def update(
        self,
        frameNum,
        img,
        current,
        N,
    ):
        next_gen = current.copy()
        # VVVV CALCS TOTAL NEIGHBORS!! VVVVV
        neighbors = [
            [
                sum(
                    [
                        1
                        if 0 < r + direction[0] < N
                        and 0 <= c + direction[1] < N
                        and current[r + direction[0]][c + direction[1]] >= 200
                        else 0
                        for direction in self.directions
                    ]
                )
                for c in range(N)
            ]
            for r in range(N)
        ]
        for r in range(N):
            for c in range(N):
                if current[r][c] >= 200 and current[r][c] <= 254:  # ALIVE
                    if neighbors[r][c] <= 3 and neighbors[r][c] >= 2:  # stay alive
                        next_gen[r][c] = current[r][c] + 1
                    else:  # decay
                        next_gen[r][c] = self.phase_set
                elif current[r][c] == 255:  # explode
                    for direction in self.directions:
                        if 0 < r + direction[0] < N and 0 <= c + direction[1] < N:
                            next_gen[r + direction[0]][c + direction[1]] = 200
                else:  # DEAD
                    if neighbors[r][c] == 3:  # back to life
                        next_gen[r][c] = 200
                    else:  # next decay
                        next_gen[r][c] = self.phases[current[r][c]]

        img.set_data(next_gen)
        current[:] = next_gen[:]
        return (img,)

    def reference_cmap(self, color_map):
        if color_map in self.color_maps:
            return color_map
        # maybe lowercase on accident
        if color_map in self.color_maps_lower:
            return self.color_maps[self.color_maps_lower[color_map]]
        # maybe uppercase on accident
        if color_map in self.color_maps_upper:
            return self.color_maps[self.color_maps_upper[color_map]]
        # they messed up big time!
        return "twilight_shifted_r"

    def gen_life_gif(self, size, update_time, color_map, interp, show=False):
        N = size if size < 101 else 100
        updateInterval = update_time
        cm = self.reference_cmap(color_map)
        it = interp if interp in self.interps else "none"
        grid = np.array([])
        grid = self.randomGrid(N)
        fig, ax = plt.subplots()
        fig.tight_layout()
        fig.set_tight_layout(True)
        ax.set_axis_off()
        img1 = ax.imshow(grid, interpolation=it)
        img1.set_cmap(cm)
        writer = animation.PillowWriter(fps=20)
        ani1 = animation.FuncAnimation(
            fig,
            self.update,
            fargs=(
                img1,
                grid,
                N,
            ),
            frames=120,
            interval=updateInterval,
        )
        if show:
            plt.show()
        else:
            ani1.save("./BOT/_utils/_gif/tmp.gif", writer=writer)
