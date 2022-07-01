import numpy as np
import multiprocessing
import argparse
import itertools

"""
Basic idea:

Given a dataset in $d$ dimensions, a function is needed that checks if this dataset can be shattered. I don't really see a way to do this other than to brute force every possible decision stump, and check if at the end the total number of classifications = $2^N$. This should scale linearly
"""

def can_shatter(dataset):
    d = len(dataset[0])
    max_labelings = 2**len(dataset)
    # both are arrays of size 2
    min_values = np.min(dataset, axis=0)
    max_values = np.max(dataset, axis=0)

    found_labels = set()
    # iterate over dimension, lower_bound, upper_bound
    iterator = zip(range(d), min_values, max_values)
    for stump_dimension, start, end in iterator:
        for stump_boundary in range(start-1, end+1):
            # anything lower than the stump boundary gets True
            labeled = dataset[:, stump_dimension] <= stump_boundary
            found_labels.add(tuple(labeled))
            # inverse is always achievable
            found_labels.add(tuple(~labeled))
            # stop when we found the max number of possible labels:
            # dataset can be shattered
            if len(found_labels) == max_labelings:
                return True
    return False

def generate_random_data(d, n):
    while True:
        yield np.random.randint(-8*n, 8*n, (n, d))

def enumerate_data(d, n):
    datasets = itertools.product(*tuple([itertools.product(*tuple([range(0, n) for _ in range(d)])) for _ in range(n)]))
    for dataset in datasets:
        yield np.array(dataset)

def check_dimensionality(config):
    d, upper_bound, max_attempts = config
    # check from highest possible VD dimension first
    # this was if it is found we can immediately stop
    iterator = list(range(1, upper_bound+1))[::-1]
    for N in iterator:
        attempts = 0
        for data in generate_random_data(d, N):
            # just need to find a single dataset that can be shattered
            if can_shatter(data):
                print(f'Highest N for dimension {d} is: {N}')
                return
            attempts += 1
            if attempts >= max_attempts:
                break
    

def main():
    parser = argparse.ArgumentParser(description='Execution Details')
    parser.add_argument('--num_processes', dest='num_processes', type=int, default=4,
                help='Number of parallel processes to use')
    parser.add_argument('--max_attempts', dest='max_attempts', type=int, default=500,
                help='Number of parallel processes to use')
    args = parser.parse_args()

    # quickly thrown together multiprocessing so I don't have to wait too long for results
    upper_bounds = [(2,4), (4,5), (6,6), (7,6), (8,6), (9,6), (10,7), (11,7), (12,7)]
    configs = [(d, upper_bound, args.max_attempts) for d,upper_bound in (upper_bounds)]
    pool_obj = multiprocessing.Pool(processes=args.num_processes)
    pool_obj.map(check_dimensionality, configs)

if __name__ == '__main__':
    main()