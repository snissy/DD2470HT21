import numpy as np
from scipy import interpolate
from PIL import Image
from matplotlib import pyplot as plt
import math as mt
from scipy import ndimage
import timeit
from tqdm import tqdm


class Texture:

    def __init__(self, textureName):
        self.map = np.array(Image.open(textureName)).mean(axis=2)

        # Normalizing the value in p(i,j) e [0, 1]
        self.map = (self.map - np.min(self.map)) / np.ptp(self.map)
        self.dims = self.map.shape

    def getTextureCords(self, point):
        x = mt.floor(point.x * self.dims[1])
        y = mt.floor(point.y * self.dims[0])
        return min(max(y, 0), self.dims[0] - 1), min(max(x, 0), self.dims[1] - 1)

    def sampleWholeSegment(self, segment):
        # We let Q be a bit extend in order to find more population
        subDir = segment.getDirectionVector()
        subDir.scale(2)
        subQ = segment.p.addVector(subDir)

        pIndex = self.getTextureCords(segment.p)
        qIndex = self.getTextureCords(subQ)
        halfIndex = ((pIndex[0] + qIndex[0]) // 2, (pIndex[1] + qIndex[1]) // 2)

        return (self.map[pIndex] * (0.5 / 6) + self.map[halfIndex] * (1.5 / 6) + self.map[qIndex] * (4 / 6)) / 3.0

    def sampleOnRoadEnd(self, segment):
        # We only check the end of the road segment.
        qY, qX = self.getTextureCords(segment.q)

        return self.map[qY][qX]


def sampleFunction(data, num):
    x0, y0 = 93, 90.8   # These are in _pixel_ coordinates!!
    x1, y1 = 194.6, 151.7  # These are in _pixel_ coordinates!!
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    t = 1/(np.linspace(10, 1, num))

    zi = ndimage.map_coordinates(data, np.vstack((y, x)))

    zTestOnPoint = ndimage.map_coordinates(data, [[0.5], [0.5]])[0]
    ziMean = np.mean(zi)

    ziSub = zi*t
    ziSubMean = np.mean(ziSub)

    fig, axes = plt.subplots(nrows=2)
    axes[0].imshow(data)
    axes[0].plot([x0, x1], [y0, y1], 'ro-')
    axes[0].axis('image')

    #plt.ylim([0, 1.2])
    axes[1].plot(zi, 'ro-', c="#134dab")
    axes[1].plot(np.ones(num) * ziMean,':', c="#5777ab")
    axes[1].plot(ziSub, 'ro-', c="#db9112")
    axes[1].plot(np.ones(num) * ziSubMean,':', c="#dbb169")
    axes[1].plot(t, 'ro-', c="#db1212")

    plt.show()
    # Extract the values along the line, using cubic interpolation
    return zi


def sampleFunctionSimple(data):
    x0, y0 = 40.2, 319.8  # These are in _pixel_ coordinates!!
    x1, y1 = 194.6, 151.7

    xh, yh = (x0 + x1) / 2, (y0 + y1) / 2

    x = np.array([x0, xh, x1])
    y = np.array([y0, yh, y1])

    # Extract the values along the line, using cubic interpolation
    zi = ndimage.map_coordinates(data, np.vstack((y, x)))
    return zi


if __name__ == '__main__':
    # construct interpolation function
    # (assuming your data is in the 2-d array "data")

    popMap = Texture("../textures/noise/simplex.png")
    d = popMap.map

    # n = 100
    # nFuncTest = 250
    # nTimeRegular = []
    # nTimeSimple = []
    #
    # for i in tqdm(range(n)):
    #     nTimeRegular.append(timeit.timeit(lambda: sampleFunction(d, i), number=nFuncTest) / nFuncTest)
    #     nTimeSimple.append(timeit.timeit(lambda: sampleFunctionSimple(d), number=nFuncTest) / nFuncTest)
    #
    # plt.plot(nTimeRegular, c="red")
    # plt.plot(nTimeSimple, c="blue")
    # plt.show()

    sampleFunction(d, 25)


