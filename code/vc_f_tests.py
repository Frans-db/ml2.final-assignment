import numpy as np

from vc_f import can_shatter, generate_random_data

def main():
    simple()
    fail_d2_n3()
    pass_d2_n3()
    random_d2_n3()

def simple():
    for d in range(1, 3):
        data = np.array([np.full(d, 0), np.full(d, 1)])
        assert can_shatter(data), f'Failed with {d}'

def fail_d2_n3():
    """
    3 points can be shattered in 2 dimensions, but not in this case
    """
    data = np.array([[0, 0], [0, 1], [0, 2]])
    assert not can_shatter(data)

def pass_d2_n3():
    """
    3 points can be shattered in 2 dimensions
    """
    data = np.array([[0, 0], [1, 1], [2, -1]])
    assert can_shatter(data)

def random_d2_n3():
    """
    Not a great test, while statistically unlikely this could in theory fail
    """
    attempts = 1_000
    succeeded = False
    for _ in range(attempts):
        data = generate_random_data(2, 3)
        if can_shatter(data):
            succeeded = True
    assert succeeded

if __name__ == '__main__':
    main()