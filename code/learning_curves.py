import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

R = 1
WIDTH = 2

def get_random_samples(n):
    return np.random.uniform(size=(n,2)) * WIDTH - (WIDTH / 2)

def get_donut_samples(n, scale=0.05):
    r = np.random.normal(loc=R, scale=scale, size=(n))
    xs = np.random.uniform(low=0, high=r, size=(n))
    ys = r - xs

    points = np.array([[x,y] for (x,y) in zip(xs, ys)])
    return points
    

def find_smallest_bounding_box_area(dataset):
    distances = np.sum(np.abs(dataset), axis=1)
    filtered = distances[distances <= R]
    if len(filtered) == 0:
        return 0
    biggest_distance = np.max(filtered)
    area =  (2*biggest_distance)**2 / 2
    return area

def main():
    trials = 200
    max_samples = 2_000

    results = []

    samples_to_test = range(2, max_samples)
    for num_samples in tqdm(samples_to_test):
        results.append([0, 0, 0])
        for _ in range(trials + 1):
            samples = get_random_samples(num_samples)
            random_area = find_smallest_bounding_box_area(samples)
            error = 2 - random_area
            results[-1][0] += (error / trials)

            samples = get_donut_samples(num_samples)
            donut_area = find_smallest_bounding_box_area(samples)
            error = 2 - donut_area
            results[-1][1] += (error / trials)

            samples = get_donut_samples(num_samples, scale=7.5)
            donut_area = find_smallest_bounding_box_area(samples)
            error = 2 - donut_area
            results[-1][2] += (error / trials)


    random_results = [x[0] for x in results]
    donut_results = [x[1] for x in results]
    bad_donut_results = [x[2] for x in results]
    plt.figure(figsize=(8, 5))
    plt.plot(samples_to_test, random_results, label='random')
    plt.plot(samples_to_test, donut_results, label='good sampling')
    plt.plot(samples_to_test, bad_donut_results, label='bad sampling')
    plt.title('Learning Curve')
    plt.xlabel('Number of Points')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()

    # samples = get_random_samples(100)
    # area = find_smallest_bounding_box_area(samples)
    # xs = [p[0] for p in samples]
    # ys = [p[1] for p in samples]
    # plt.scatter(xs, ys)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(f'{area}')
    # plt.show()