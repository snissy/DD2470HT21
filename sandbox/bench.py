import time as tm
from tqdm import tqdm
from multiprocessing import pool


def countFunc(test):
    sum = 0
    for i in range(100000):
        sum += i

    return sum


if __name__ == '__main__':
    startT = tm.time()

    p = pool.Pool()

    p.map(countFunc, range(2000))

    print(tm.time() - startT)

    # Single thread

    # Nils: 11.4
    # Elliot 7.8

    # Multi thread

    # Nils 8.361
    # Elliot 2.11
