from math import log2


dimensionalities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1024, 2**100]

for d in dimensionalities:
    best = -1
    for N in range(1_000):
        if 2*N*d >= 2**N:
            best = N
    print(d, best, 2* (log2(d) + 1))