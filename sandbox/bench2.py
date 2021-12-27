import time

if __name__ == '__main__':

    print("Staring processes")

    startTime = time.time()
    n = 1000000
    theRange = range(n)
    nTimes = 1000
    suma = 0

    for ii in range(nTimes):
        suma = sum(theRange)

        suma = 0

    print(suma)
    print(time.time() - startTime)
    # 00:00:02.1478550 C#
    # 00:00:23.7082381 python
    # 00:00:2.467 c++
    #

    """for (int ii= 0; ii < nTimes; ii++)
        {
        for (int i = 0; i < n; i++)
        {
            sum += i;

        }
        sum = 0;
        }"""
