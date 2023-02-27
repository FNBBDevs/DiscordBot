import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


color_maps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
interps = ['antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman']
show_cons = False
A = 200
D1 = 190
D2 = 180
D3 = 170
D4 = 160
D5 = 150
D6 = 140
D7 = 130
D8 = 120
D9 = 110
D10 = 100
D11 = 90
D12 = 80
D13 = 70
D14 = 60
D15 = 50
D16 = 40
D17 = 30
D18 = 20
D19 = 10
DEAD = 0

def randomGrid(N):
    """ this function generates an NxN list of random values """
    return np.random.choice([D1, A], N*N, ).reshape(N, N)

def update(frameNum, img, current, N,):
    counter = 0
    next_gen = current.copy()
    for r in range(N):
        for c in range(N):
            total = 0
            if r > 0 and r < N-1 and c > 0 and c < N-1:
                if A<=current[r-1][c-1]<=255:
                    total += 1
                if A<=current[r-1][c]<=255:
                    total += 1
                if A<=current[r-1][c+1]<=255:
                    total+=1
                if A<=current[r][c-1]<=255:
                    total+=1
                if A<=current[r][c+1]<=255:
                    total+=1
                if A<=current[r+1][c-1]<=255:
                    total += 1
                if A<=current[r+1][c]<=255:
                    total += 1
                if A<=current[r+1][c+1]<=255:
                    total+=1
            elif r == 0 and c > 0 and c < N-1:
                if A<=current[r][c-1]<=255:
                    total += 1
                if A<=current[r][c+1]<=255:
                    total += 1
                if A<=current[r+1][c-1]<=255:
                    total+=1
                if A<=current[r+1][c]<=255:
                    total+=1
                if A<=current[r+1][c+1]<=255:
                    total += 1
            elif r == N-1 and c > 0 and c < N-1:
                if A<=current[r-1][c-1]<=255:
                    total += 1
                if A<=current[r-1][c]<=255:
                    total += 1
                if A<=current[r-1][c+1]<=255:
                    total+=1
                if A<=current[r][c-1]<=255:
                    total +=1
                if A<=current[r][c+1]<=255:
                    total+=1
            elif r > 0 and r < N-1 and c == 0:
                if A<=current[r-1][c]<=255:
                    total+=1
                if A<=current[r-1][c+1]<=255:
                    total+=1
                if A<=current[r][c+1]<=255:
                    total+=1
                if A<=current[r+1][c]<=255:
                    total+=1
                if A<=current[r+1][c+1]<=255:
                    total+=1
            elif r > 0 and r < N-1 and c == N-1:
                if A<=current[r-1][c-1]<=255:
                    total += 1
                if A<=current[r-1][c]<=255:
                    total += 1
                if A<=current[r][c-1]<=255:
                    total += 1
                if A<=current[r+1][c-1]<=255:
                    total += 1
                if A<=current[r+1][c]<=255:
                    total += 1
            elif r == 0 and c == 0:
                if A<=current[r][c+1]<=255:
                    total += 1
                if A<=current[r+1][c]<=255:
                    total += 1
                if A<=current[r+1][c+1]<=255:
                    total += 1
            elif r == 0 and c == N - 1:
                if A<=current[r][c-1]<=255:
                    total += 1
                if A<=current[r+1][c-1]<=255:
                    total += 1
                if A<=current[r+1][c]<=255:
                    total += 1
            elif r == N-1 and c == 0:
                if A<=current[r-1][c]<=255:
                    total += 1
                if A<=current[r-1][c+1]<=255:
                    total += 1
                if A<=current[r][c+1]<=255:
                    total += 1
            elif r == N-1 and c == N-1:
                if A<=current[r][c-1]<=255:
                    total += 1
                if A<=current[r-1][c-1]<=255:
                    total += 1
                if A<=current[r-1][c]<=255:
                    total += 1
            if 200 <= current[r][c]<=255: # Alive
                if total > 3 or total < 2: # dead
                    next_gen[r][c] = D1
                elif current[r][c] != 255:
                    next_gen[r][c] = current[r][c] + 1
                elif current[r][c] == 255:
                    ayo = A
                    next_gen[r][c] = D1
                    try:
                        next_gen[r + 1][c + 1] = ayo
                    except IndexError:
                        pass
                    try:
                        next_gen[r - 1][c - 1] = ayo
                    except IndexError:
                        pass  
                    try:
                        next_gen[r + 1][c - 1] = ayo
                    except IndexError:
                        pass  
                    try:
                        next_gen[r - 1][c + 1] = ayo
                    except IndexError:
                        pass 
                    try:
                        next_gen[r][c + 1] = ayo
                    except IndexError:
                        pass  
                    try:
                        next_gen[r + 1][c] = ayo
                    except IndexError:
                        pass  
                    try:
                        next_gen[r][c - 1] = ayo
                    except IndexError:
                        pass 
                    try:
                        next_gen[r][c - 1] = ayo
                    except IndexError:
                        pass   
            else: # Dead
                if total == 3:
                    next_gen[r][c] = A
                else:
                    val = current[r][c]
                    if val == D1:
                        next_gen[r][c] = D2
                    elif val == D2:
                        next_gen[r][c] = D3
                    elif val == D3:
                        next_gen[r][c] = D4
                    elif val == D4:
                        next_gen[r][c] = D5
                    elif val == D5:
                        next_gen[r][c] = D6
                    elif val == D6:
                        next_gen[r][c] = D7
                    elif val == D7:
                        next_gen[r][c] = D8
                    elif val == D8:
                        next_gen[r][c] = D9
                    elif val == D9:
                        next_gen[r][c] = D10
                    elif val == D10:
                        next_gen[r][c] = D11
                    elif val == D11:
                        next_gen[r][c] = D12
                    elif val == D12:
                        next_gen[r][c] = D13
                    elif val == D13:
                        next_gen[r][c] = D14
                    elif val == D14:
                        next_gen[r][c] = D15
                    elif val == D15:
                        next_gen[r][c] = D16
                    elif val == D16:
                        next_gen[r][c] = D17
                    elif val == D17:
                        next_gen[r][c] = D18
                    elif val == D18:
                        next_gen[r][c] = D19
                    elif val == D19:
                        next_gen[r][c] = DEAD
                if next_gen[r][c] == DEAD:
                    counter += 1
        
    if counter == N*N:
        next_gen = randomGrid(N)

    img.set_data(next_gen)
    current[:] = next_gen[:]
    return img,


def gen_life_gif(size, update_time, color_map, interp):
    print('in gen life gif')
    N = size if size < 501 else 500
    updateInterval = update_time
    cm = color_map if color_map in color_maps else "tab20b"
    it = interp if interp in interps else "none"
    grid = np.array([])
    grid = randomGrid(N)
    plt.tight_layout()
    fig, ax = plt.subplots()
    ax.set_axis_off()
    img1 = ax.imshow(grid, interpolation=it)
    img1.set_cmap(cm)
    writer = animation.PillowWriter(fps=20)
    print('running animation')
    ani1 = animation.FuncAnimation(fig, update, fargs=(img1, grid, N, ),
                                frames = 120,
                                interval=updateInterval,
                                save_count=50)
    print('saving animation')
    ani1.save("./BOT/_utils/_gif/tmp.gif", writer=writer)
    print('done saving animation')
