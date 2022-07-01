import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

R = 1
WIDTH = 2

def get_random_samples(n):
    return np.random.normal(size=(n, 2), loc=0, scale=0.1)

def get_donut_samples(n, scale=0.2):
    xs = np.random.uniform(-R, R, size=(n))
    ys = np.full((n, ), 0) - (xs)
    # sampled_r = np.random.normal(loc=R, scale=scale, size=(n))
    # deg = np.random.uniform(0, 2*np.pi, size=(n))
    # xs = sampled_r * np.cos(deg)
    # ys = sampled_r * np.sin(deg)
    points = np.array([[x,y] for (x,y) in zip(xs, ys)])
    return points

def find_smallest_bounding_box_area(dataset):
    print(dataset)
    distances = np.sum(np.abs(dataset), axis=1)
    print(distances)
    filtered = dataset[distances <= R]
    print(filtered)
    if len(filtered) == 0:
        return 0
    lower_bound = np.min(filtered, axis=0)
    upper_bound = np.max(filtered, axis=0)
    print('---')
    print(lower_bound)
    print(upper_bound)
    lengths = upper_bound - lower_bound
    print(lengths)
    area = np.prod(lengths)
    return area / 2

def main():
    trials = 10
    max_samples = 100
    random_results = []
    donut_results = []
    samples_to_test = range(2, max_samples)
    for num_samples in tqdm(samples_to_test):
        random_result = 0
        donut_result = 0
        for _ in range(trials + 1):
            samples = get_random_samples(num_samples)
            random_area = find_smallest_bounding_box_area(samples)
            random_result += (random_area / trials)

            samples = get_random_samples(num_samples)
            donut_area = find_smallest_bounding_box_area(samples)
            donut_result += (donut_area / trials)
        random_results.append(random_result)
        donut_results.append(donut_result)

    plt.figure(figsize=(8, 5))
    plt.plot(samples_to_test, random_results, label='random')
    plt.plot(samples_to_test, donut_results, label='donut')
    plt.title('Learning Curve - Donut Distribution')
    plt.xlabel('Number of Points')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    # main()
    samples = get_donut_samples(1_000)
    area = find_smallest_bounding_box_area(samples)
    print(area)
    xs = [p[0] for p in samples]
    ys = [p[1] for p in samples]
    plt.scatter(xs, ys)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Donut distribution - m = 10000')
    plt.show()

    # samples = get_random_samples(100)
    # area = find_smallest_bounding_box_area(samples)
    # xs = [p[0] for p in samples]
    # ys = [p[1] for p in samples]
    # plt.scatter(xs, ys)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(f'{area}')
    # plt.show()