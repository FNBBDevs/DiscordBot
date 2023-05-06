current = [[np.random.choice([0,1]) for _ in range(10)] for __ in range(10)]
next_gen = [[0 for _ in range(10)] for __ in range(10)]
for i in range(4):
    for r in range(10):
        for c in range(10):
            neighbors = 0
            for d1, d2 in [(0,1),(1,0),(-1,0),(0,-1),(1,1),(-1,-1),(1,-1),(1,-1)]:
                if r + d1 >= 0 and r + d1 < 10 and c + d2 >= 0 and c + d2 < 10 and current[r + d1][c + d2] == 1:
                    neighbors += 1
            if current[r][c] == 1:
                if neighbors > 1 and neighbors < 4:
                    next_gen[r][c] = 1
                else:
                    next_gen[r][c] = 0
            else:
                if neighbors == 3:
                    next_gen[r][c] = 1
                else:
                    next_gen[r][c] = 0
    current[:] = next_gen[:]
    print('-'*12)
    for row in current:
        print('|', end='')
        for col in row:
            if col == 1:
                print("o", end="")
            else:
                print(" ", end="")
        print('|', end='')
        print()
    print('-'*12)
    print()