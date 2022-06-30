import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

r = 1
width = 2



def get_random_samples(n):
    return np.random.random((n, 2)) * width - 1

def get_donut_samples(n, loc=2**(-0.5), scale=0.5):
    r = np.random.normal(loc=loc, scale=scale, size=(n))
    deg = np.random.uniform(0, 2*np.pi, size=(n))
    xs = r * np.cos(deg)
    ys = r * np.sin(deg)
    points = np.array([[x,y] for (x,y) in zip(xs, ys)])
    return points

def find_smallest_bounding_box_area(dataset):
    distances = np.sum(np.abs(dataset), axis=1)
    filtered = dataset[distances <= r]
    if len(filtered) == 0:
        return 0
    lower_bound = np.min(filtered, axis=0)
    upper_bound = np.max(filtered, axis=0)
    lengths = upper_bound - lower_bound
    area = np.prod(lengths)
    return area / 2

def main():
    trials = 100
    max_samples = 10_000
    results = []
    samples_to_test = range(2, max_samples)
    for num_samples in tqdm(samples_to_test):
        trial_result = 0
        for _ in range(trials):
            samples = get_random_samples(num_samples)
            area = find_smallest_bounding_box_area(samples)
            trial_result += (area / trials)
        error = (2 - trial_result) / 2
        results.append(error)

    plt.plot(samples_to_test, results)
    plt.title('Learning Curve - Normal Distribution N(0, 1)')
    plt.xlabel('Number of Points')
    plt.ylabel('Error')
    plt.show()

if __name__ == '__main__':
    main()
    # points = get_donut_samples(10_000)
    # xs = [p[0] for p in points]
    # ys = [p[1] for p in points]
    # plt.scatter(xs, ys)
    # plt.show()