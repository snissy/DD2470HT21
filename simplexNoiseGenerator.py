import numpy as np
import random as rd
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
The code for this file will mostly be based on this blogpost by Stefan Gustavson
https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
"""


def floor(x):
    xi = int(x)
    return xi - 1 if x < xi else xi
    # x<xi ? xi-1 : xi;


class Vector2D:
    # For 2D, 8 or 16

    cornerVectors = [(np.cos(i), np.sin(i)) for i in np.linspace(0, 2 * np.pi, 16)]

    @staticmethod
    def get_random_GradientComponent():
        return np.array([*rd.choice(Vector2D.cornerVectors)])


class LatticePoints:

    def __init__(self):
        # The current set up works well but want to try this as well
        self.lattices_points = {}

    def getPoint(self, r, c):
        key = (r, c)
        if key in self.lattices_points:
            return self.lattices_points[key]

        else:
            rd_gradient = Vector2D.get_random_GradientComponent()
            self.lattices_points[key] = rd_gradient
            return rd_gradient


class SimplexNoiseGrid:
    F2 = 0.5 * (np.sqrt(3.0) - 1.0)  # Here we create a factor in order to go from our triangle grid to quad grid.
    G2 = (3.0 - np.sqrt(3.0)) / 6.0

    def __init__(self, pWidth, pHeight, sFreq=2, sAmp=1, n_octaves=4, customFreqAmp=None):
        self.pWidth = pWidth
        self.pHeight = pHeight
        self.data = np.zeros((pHeight, pWidth))

        self.pWidth_range = range(pWidth)
        self.pHeight_range = range(pHeight)

        diagonal = (pWidth ** 2 + pHeight ** 2) ** (1 / 2)

        if customFreqAmp:
            self.freq_amp = customFreqAmp
        else:
            self.freq_amp = [(int(diagonal / (sFreq * 2 ** i)), sAmp * (2 / (2 ** (i + 1)))) for i in range(n_octaves)]

        self.lattices_points = None
        self.generateData()

    def generateData(self):
        for f, amp in tqdm(self.freq_amp):
            self.lattices_points = LatticePoints()
            self.data += np.array(
                [[self.__calcPoint(r / f, c / f) * amp for c in self.pWidth_range] for r in
                 self.pHeight_range])

    def __calcPoint(self, yin, xin):

        # Skew the input space to determine which simplex cell we're in
        s = (xin + yin) * self.F2
        i = floor(xin + s)
        j = floor(yin + s)
        t_factor = (i + j) * self.G2
        X0 = i - t_factor
        Y0 = j - t_factor
        x0 = xin - X0
        y0 = yin - Y0

        if x0 > y0:
            i1 = 1
            j1 = 0
        else:
            i1 = 0
            j1 = 1

        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2

        gi0 = self.lattices_points.getPoint(j, i)
        gi1 = self.lattices_points.getPoint(j + j1, i + i1)
        gi2 = self.lattices_points.getPoint(j + 1, i + 1)

        t0 = 0.5 - x0 * x0 - y0 * y0
        if t0 < 0:
            n0 = 0.0
        else:
            t0 *= t0
            n0 = t0 * t0 * gi0.dot(np.array([x0, y0]))

        t1 = 0.5 - x1 * x1 - y1 * y1
        if t1 < 0:
            n1 = 0.0
        else:
            t1 *= t1
            n1 = t1 * t1 * gi1.dot(np.array([x1, y1]))

        t2 = 0.5 - x2 * x2 - y2 * y2
        if t2 < 0:
            n2 = 0.0
        else:
            t2 *= t2
            n2 = t2 * t2 * gi2.dot(np.array([x2, y2]))

        return 70.0 * (n0 + n1 + n2)

    def getData(self):
        return self.data


if __name__ == '__main__':
    t = SimplexNoiseGrid(100, 100)
    data = t.getData()

    # Normalised [0,1]
    data = (data - np.min(data)) / np.ptp(data)
    plt.imshow(data, cmap="gray")

    # data = np.exp(data-1)*np.abs(np.sin(data*2*np.pi*8))
    # data = 1 - np.power((np.cos(data*np.pi*2*20)+1)/2, 5)
    #data = 1 - np.power((np.cos(np.pi + data*np.pi*50)+1)/2, 100)
    #plt.imshow(data, cmap="gray")

    plt.axis('off')
    plt.savefig("textures/noise/simplex.png", bbox_inches='tight', transparent=True, pad_inches=0)

    plt.show()
