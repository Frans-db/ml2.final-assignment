import numpy as np
import multiprocessing
import argparse

"""
Basic idea:

Given a dataset in $d$ dimensions, a function is needed that checks if this dataset can be shattered. I don't really see a way to do this other than to brute force every possible decision stump, and check if at the end the total number of classifications = $2^N$. This should scale linearly
"""

def can_shatter(dataset):
    d = len(dataset[0])
    max_labelings = 2**len(dataset)
    min_values = np.min(dataset, axis=0)
    max_values = np.max(dataset, axis=0)

    found_labels = set()


    for stump_dimension, start, end in zip(range(d), min_values, max_values):
        for stump_boundary in range(start-1, end+1):
            labeled = dataset[:, stump_dimension] <= stump_boundary
            found_labels.add(tuple(labeled))
            found_labels.add(tuple(~labeled)) # inverse is always achievable

            if len(found_labels) == max_labelings:
                return True
    return False

def generate_random_data(d, n):
    return np.random.randint(0, n, (n, d))

def check_dimensionality(config):
    d, upper_bound, max_attempts = config
    highest_N = -1
    for N in range(1, upper_bound+1):
        for _ in range(max_attempts):
            data = generate_random_data(d, N)
            if can_shatter(data):
                highest_N = N
    print(f'Highest N for dimension {d} is: {highest_N}')

def main():
    parser = argparse.ArgumentParser(description='Execution Details')
    parser.add_argument('--num_processes', dest='num_processes', type=int, default=4,
                help='Number of parallel processes to use')
    parser.add_argument('--max_attempts', dest='max_attempts', type=int, default=100_000,
                help='Number of parallel processes to use')
    args = parser.parse_args()

    # quickly thrown together multiprocessing so I don't have to wait too long for results
    upper_bounds = [2, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7]
    configs = [(d+1, upper_bound, args.max_attempts) for d,upper_bound in enumerate(upper_bounds)]
    pool_obj = multiprocessing.Pool(processes=args.num_processes)
    pool_obj.map(check_dimensionality, configs)

if __name__ == '__main__':
    main()