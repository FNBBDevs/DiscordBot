no x = 0
no balls_count = 0
no cock_count = 0
bounce
    if random() < 0.5
        nolout("cock")
        cock_count = cock_count + 1
    hermph
        nolout("balls")
        balls_count = balls_count + 1
    x = x + 1
while x < 100
nolout("Total Balls: " + balls_count)
nolout("Total Cocks: " + cock_count)