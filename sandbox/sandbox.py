import random as rd
import numpy as np
from matplotlib import pyplot as plt

nCol = 50
nRow = 25

for i in range(15):
    data = np.ones((nRow, nCol))

    for r in range(nRow):

        for c in range(nCol):
            data[r, c] = round((c+1) * rd.gauss(1.25 * c, (c+1)*0.15), 4)

    plt.plot(list(range(1, nCol+1)), np.median(data, axis=0))
    #plt.plot(list(range(1, nCol+1)), np.max(data, axis=0))
    #plt.plot(list(range(1, nCol+1)), np.min(data, axis=0))
#plt.boxplot(data, showfliers = False)

plt.plot(list(range(1, nCol+1)), np.ones(nCol)*(500), 'r.')
plt.show()
