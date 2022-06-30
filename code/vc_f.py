from cProfile import label
import numpy as np

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

def main():
    pass

data = np.array([[0], [1]]) # can be shattered in any dimension
if __name__ == '__main__':
    pass