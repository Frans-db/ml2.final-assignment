from math import comb, floor

for d in range(1, 13):
    for m in range(2, 15):
        num = comb(m, floor(m / 2))
        if 2*d < num:
            print(f'VC-dimension for {d}: {m-1}')
            break