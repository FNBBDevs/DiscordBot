import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


color_maps = plt.colormaps()
color_maps_lower = {cmap.lower():i for i, cmap in enumerate(color_maps)}
color_maps_upper =  {cmap.upper():i for i, cmap in enumerate(color_maps)}
interps = ['antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman']
directions = [[1, 0],[0, 1],[-1, 0],[0, -1],[1, 1],[1, -1],[-1, 1],[-1, -1]]
phases = {200:190, 190:180, 180:170, 170:160, 160:150, 150:140, 140:130, 130:120, 120:110, 110:90, 90:80, 80:70, 70:60, 60:50, 50:40, 40:30, 30:20, 20:10, 10:0, 0:0}

def randomGrid(N):
    """ this function generates an NxN list of random values """
    return np.random.choice([200, 0], N*N, ).reshape(N, N)

def update(frameNum, img, current, N,):
    counter = 0
    next_gen = current.copy()
    # VVVV CALCS TOTAL NEIGHBORS!! VVVVV
    neighbors = [[sum([1 if 0<r+direction[0]<N and 0<= c+direction[1]<N and current[r+direction[0]][c+direction[1]]>=200 else 0 for direction in directions]) for c in range(N)] for r in range(N)]
    for r in range(N):
        for c in range(N):
            if current[r][c] >= 200 and current[r][c] <= 254: # ALIVE
                if neighbors[r][c] <= 3 and neighbors[r][c] >= 2: #stay alive
                    next_gen[r][c] = current[r][c] + 1
                else: # decay
                    next_gen[r][c] = 190
            elif current[r][c] == 255: # explode
                for direction in directions:
                    if 0<r+direction[0]<N and 0<= c+direction[1]<N:
                        next_gen[r+direction[0]][c+direction[1]] = 200
            else: # DEAD
                if neighbors[r][c] == 3: # back to life
                    next_gen[r][c] = 200
                else: # next decay
                    next_gen[r][c] = phases[current[r][c]]

    img.set_data(next_gen)
    current[:] = next_gen[:]
    return img,


def reference_cmap(color_map):
    if color_map in color_maps:
        return color_map
    # maybe lowercase on accident
    if color_map in color_maps_lower:
        return color_maps[color_maps_lower[color_map]]
    # maybe uppercase on accident
    if color_map in color_maps_upper:
        return color_maps[color_maps_upper[color_map]]
    # they messed up big time!
    return "twilight_shifted_r"



def gen_life_gif(size, update_time, color_map, interp):
    N = size if size < 151 else 150
    updateInterval = update_time
    cm = reference_cmap(color_map)
    it = interp if interp in interps else "none"
    grid = np.array([])
    grid = randomGrid(N)
    fig, ax = plt.subplots()
    fig.tight_layout()
    fig.set_tight_layout(True)
    ax.set_axis_off()
    img1 = ax.imshow(grid, interpolation=it)
    img1.set_cmap(cm)
    writer = animation.PillowWriter(fps=20)
    ani1 = animation.FuncAnimation(fig, update, fargs=(img1, grid, N, ),
                                frames = 120,
                                interval=updateInterval,
                                save_count=50)
    ani1.save("./BOT/_utils/_gif/tmp.gif", writer=writer)
