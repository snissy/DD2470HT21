import time

import numpy as np
import random as rd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.stats import norm
import math as mt
from skimage.transform import radon, iradon

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
        for f, amp in self.freq_amp:
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


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def DataOnlyPositiv(data):
    return data - np.min(data)


def setSum_toOne(data):
    return data / np.sum(data)


def oneDimensionWasserSteain(n):
    res = []
    rd.seed(int(time.time()))

    seed = rd.randint(0, 10000000000000000000)
    rd.seed(seed)

    for i in tqdm(range(n)):
        seed += 1
        rd.seed(seed)
        tV = SimplexNoiseGrid(1, 50).getData()
        tU = SimplexNoiseGrid(1, 50).getData()

        testV = NormalizeData(tV[:, 0])
        cdfTESTV = np.cumsum(testV)
        cdfTESTV *= (1 / np.max(cdfTESTV))

        testU = NormalizeData(tU[:, 0])
        cdfTESTU = np.cumsum(testU)
        cdfTESTU *= (1 / np.max(cdfTESTU))

        wDistance = wasserstein_distance(cdfTESTV, cdfTESTU)

        res.append((wDistance, testV, testU))

    wDistance, testV, testU = max(res, key=lambda e: e[0])

    cdfTESTV = np.cumsum(testV)
    cdfTESTV *= (1 / np.max(cdfTESTV))
    cdfTESTU = np.cumsum(testU)
    cdfTESTU *= (1 / np.max(cdfTESTU))

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Wasserstein distance:  {}'.format(round(wDistance, 5)), fontsize=16)

    ax1.plot(testV, c="blue")
    ax1.plot(testU, c="red")

    ax2.plot(cdfTESTV, c="blue")
    ax2.plot(cdfTESTU, c="red")
    plt.show()

    wDistance, testV, testU = min(res, key=lambda e: e[0])
    cdfTESTV = np.cumsum(testV)
    cdfTESTV *= (1 / np.max(cdfTESTV))
    cdfTESTU = np.cumsum(testU)
    cdfTESTU *= (1 / np.max(cdfTESTU))

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Wasserstein distance:  {}'.format(round(wDistance, 5)), fontsize=16)

    ax1.plot(testV, c="blue")
    ax1.plot(testU, c="red")

    ax2.plot(cdfTESTV, c="blue")
    ax2.plot(cdfTESTU, c="red")
    plt.show()


def twoDimesionWasserStein(n):
    res = []
    rd.seed(int(time.time()))
    seed = rd.randint(0, 10000000000000000000)
    rd.seed(seed)

    tV = SimplexNoiseGrid(45, 45).getData()
    tV = DataOnlyPositiv(tV)
    tV = tV * (1 / np.sum(tV))

    for i in tqdm(range(n)):
        seed += 1
        rd.seed(seed)
        tU = SimplexNoiseGrid(45, 45).getData()

        tU = DataOnlyPositiv(tU)
        tU = tU * (1 / np.sum(tU))

        def sliced_wasserstein(X, Y, num_proj):
            eWasserstein = []
            for _ in range(num_proj):
                # sample uniformly from the unit spher

                theta = [rd.random() * 360]

                X_proj = radon(X, theta)
                X_proj = np.cumsum(X_proj)
                X_proj *= (1 / np.max(X_proj))

                Y_proj = radon(Y, theta)
                Y_proj = np.cumsum(Y_proj)
                Y_proj *= (1 / np.max(Y_proj))

                # compute 1d wasserstein
                eWasserstein.append(wasserstein_distance(X_proj, Y_proj))
            return np.mean(eWasserstein)

        wD = sliced_wasserstein(tV, tU, 550)

        res.append((wD, tV, tU))

    wDistance, tV, tU = max(res, key=lambda e: e[0])

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Wasserstein distance:  {}'.format(round(wDistance, 5)), fontsize=16)
    print(np.sum(np.abs(tV - tU)))
    ax1.imshow(tV)
    ax2.imshow(tU)

    plt.show()

    wDistance, tV, tU = min(res, key=lambda e: e[0])

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Wasserstein distance:  {}'.format(round(wDistance, 5)), fontsize=16)

    print(np.sum(np.abs(tV - tU)))
    ax1.imshow(tV)
    ax2.imshow(tU)

    plt.show()


if __name__ == '__main__':

    #twoDimesionWasserStein(250)

    twoDimesionWasserStein(15500)
