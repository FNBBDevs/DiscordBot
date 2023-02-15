# -------------------------- ESSENTIAL FUNCITONS -------------------------------

def _check(word):
    checks = {0: 'fort',1: 'nite',2: 'balls'}
    res = [False, False, False]
    word = str(word).lower()
    for check, val in checks.items():
        cp = 0
        cc = val[cp]
        cw = ""
        for i, c in enumerate(word):
            if c == cc:
                cw += c
                cp += 1
            if cw == val:
                break
            if cp >= len(val):
                res[check] = False
            else:
                cc = val[cp]
        if cw != val:
            res[check] = False
        else:
            res[check] = True
    return res

def erm__________is_this_fortnite_balls(word):
    tmp1 = _check(word)
    tmp2 = _check(word[::-1])
    return all([(tmp1[0] or tmp2[0]),(tmp1[1] or tmp2[1]),(tmp1[2] or tmp2[2])])