import time as tm
from tqdm import tqdm
from multiprocessing import pool


def countFunc(test):
    sum = 0
    for i in range(1000):
        sum += i

    return sum


if __name__ == '__main__':
    p = pool.Pool()

    startT = tm.time()

    for i in tqdm(range(5000)):
        t = list(p.map(countFunc, range(10)))

    print("The pool Object time {}".format(tm.time() - startT, 5))
    p.close()
    startT = tm.time()
    for i in tqdm(range(5000)):
        t = list(map(countFunc, range(10)))

    print("The pool Object time {}".format(tm.time() - startT, 5))

    # Single thread

    # Nils: 11.4
    # Elliot 7.8

    # Multi thread

    # Nils 8.361
    # Elliot 2.11
