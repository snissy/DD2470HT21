from dataclasses import dataclass, field
import heapq
import math as mt
from typing import Any
import json
from PIL import Image
import numpy as np
import quads as qd
import random as rd

import imageio
from tqdm import tqdm
import cProfile as profile
import time as tm
import cv2
from matplotlib import pyplot as plt


config = json.loads(open("config.json", mode="r").read())
clamp = lambda x, l, u: l if x < l else u if x > u else x


class ExtendedPoint(qd.Point):

    def __init__(self, x, y, data=None):
        super().__init__(x, y, data)

    def distanceTo(self, other):
        return mt.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def addVector(self, other):
        newX = self.x + other.x
        newY = self.y + other.y

        return ExtendedPoint(newX, newY)

    def getMagnitude(self):
        return (self.x ** 2 + self.y ** 2) ** .5

    def setLength(self, scale):
        """
        The vector will get the length of the scale
        :param scale:
        :return:
        """

        mag = self.getMagnitude()
        if mag <= 0.000001:
            print("WARNING: You get set length on zero vector")
            raise Exception("You get set length on zero Vector")

        mag = scale / mag
        self.scale(mag)

        return self

    def scale(self, s):
        self.x *= s
        self.y *= s

    def __eq__(self, other):
        return abs(self.x - other.x) <= 0.0000001 and abs(self.y - other.y) <= 0.0000001

    def dotProduct(self, other):
        return (self.x * other.x) + (self.y * other.y)


class Segment:
    p: ExtendedPoint
    q: ExtendedPoint
    highway: bool

    def __init__(self, p, q, highway=False):

        self.p = p
        self.q = q
        self.highway = highway

    def getDirectionVector(self):

        newPoint = ExtendedPoint(self.q.x - self.p.x, self.q.y - self.p.y)
        return newPoint

    def intersection(self, other):

        """
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        self.p = (x1, y1)
        self.q = (x2, y2)
        other.p = (x3, y3)
        other.q = (x4, y4)
        :param other:
        :return:
        # Note to myself. I am never going to write this code again.
        # NEVER EVER WRITE THIS CODE AGAIN! OR USE A LIBRARY
        """

        a = self.p.x - other.p.x  # x1 - x3
        b = other.p.y - other.q.y  # y3 - y4
        c = self.p.y - other.p.y  # y1 - y3
        d = other.p.x - other.q.x  # x3 - x4
        e = self.p.x - self.q.x  # x1 - x2
        f = self.p.y - self.q.y  # y1 - y2

        determinant = (e * b - f * d)

        if abs(determinant) <= 0.000001:
            # The line parallel, this requires some more checking

            def is_between(p1, p3, p2):
                return abs((p1.distanceTo(p3) + p3.distanceTo(p2) - p1.distanceTo(p2))) < 0.000001

            if is_between(self.p, other.p, self.q):
                return other.p, 0 if self.p.distanceTo(other.p) < self.q.distanceTo(other.p) else 1, 0
            if is_between(self.p, other.q, self.q):
                return other.q, 0 if self.p.distanceTo(other.q) < self.q.distanceTo(other.q) else 1, 1

        else:
            # t for self Line
            # u for other Line
            t = (a * b - c * d) / determinant
            if 0 <= t <= 1:
                u = (a * f - c * e) / determinant

                if 0 <= u <= 1:
                    newX = self.p.x + (self.q.x - self.p.x) * t
                    newY = self.p.y + (self.q.y - self.p.y) * t
                    return ExtendedPoint(newX, newY), t, u

        return False, None, None

    def getMidPoint(self):

        midPoint = ExtendedPoint((self.p.x + self.q.x) / 2.0, (self.p.y + self.q.y) / 2.0)

        return midPoint

    def getClosestPointOnSegment(self, point):

        x1 = self.p.x
        y1 = self.p.y
        x2 = self.q.x
        y2 = self.q.y
        x3 = point.x
        y3 = point.y

        px = x2 - x1
        py = y2 - y1
        norm = px * px + py * py
        u = max(0, min(1, ((x3 - x1) * px + (y3 - y1) * py) / norm))

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        dist = (dx * dx + dy * dy) ** .5

        return dist, ExtendedPoint(x, y), u

    def __eq__(self, other):

        equals = abs(self.p.x - other.p.x) <= 0.000001
        equals = equals and abs(self.p.y - other.p.y) <= 0.000001
        equals = equals and abs(self.q.x - other.q.x) <= 0.000001

        return equals and abs(self.q.y - other.q.y) <= 0.000001


@dataclass
class Item:
    data: Any = field(compare=False)
    prioValue: float = mt.inf

    def __lt__(self, other):
        return self.prioValue < other.prioValue


class PriorityQueue:

    def __init__(self):
        self.__que = []

    def pop(self):
        """
        :return: returns the most prioritized item
        """
        return heapq.heappop(self.__que)

    def push(self, item):
        """
        :param item: The item you want to store
        :return:
        """
        heapq.heappush(self.__que, item)

    def pushAll(self, itemList):
        for item in itemList:
            self.push(item)

    def empty(self):
        return len(self.__que) == 0

    def getAllData(self):
        return [item.data for item in self.__que]


class SegmentsContainer:
    highwaySearchLength = 1 * config["highwayLength"]
    normalSearchLength = (config["highwayLength"] / 2) + (config["normalLength"] / 2)

    def __init__(self):
        self.allSegments = []
        self.tree = qd.QuadTree((0.5, 0.5), 1, 1)

    def addSegment(self, s):
        self.allSegments.append(s)
        midPoint = s.getMidPoint()
        self.tree.insert(midPoint)
        midPoint.data = s

    def __len__(self):
        return len(self.allSegments)

    def getCloseSegments(self, segment):
        # ((5)^(1/2)/2) = 1.1180 <= 1.12
        point = segment.getMidPoint()
        sLength = self.highwaySearchLength if segment.highway else self.normalSearchLength

        minX = point.x - sLength
        maxX = point.x + sLength
        minY = point.y - sLength
        maxY = point.y + sLength
        pointBB = qd.BoundingBox(minX, minY, maxX, maxY)

        res = self.tree.within_bb(pointBB)
        return res


class Texture:

    def __init__(self, textureName=None):

        if textureName:
            self.map = np.array(Image.open(textureName)).mean(axis=2)
            # Normalizing the value in p(i,j) e [0, 1]
            self.map = (self.map - np.min(self.map)) / np.ptp(self.map)
        else:
            self.map = np.ones((100, 100))

        dimY, dimX = self.map.shape
        self.dimX = dimX - 1
        self.dimY = dimY - 1

    def getTextureCords(self, point):
        x = mt.floor(point.x * self.dimX)
        y = mt.floor(point.y * self.dimY)

        x = 0 if x < 0 else self.dimX if x > self.dimX else x
        y = 0 if y < 0 else self.dimY if y > self.dimY else y

        return x, y

    def sampleWholeSegment(self, segment):
        pTx, pTy = self.getTextureCords(segment.p)
        qTx, qTy = self.getTextureCords(segment.q)
        hTx, hTy = (qTx + pTx) // 2, (qTy + pTy) // 2,

        # Unfortunately this takes too much time
        # num = self.sampleSize
        # x, y = np.linspace(pTx, qTx, num), np.linspace(pTy, qTy, num)
        # zi = np.mean(ndimage.map_coordinates(self.map, np.vstack((y, x)), mode="nearest") * self.tRange)

        return (self.map[pTy][pTx] * (0.5 / 6) + self.map[hTy][hTx] * (1 / 6) + self.map[qTy][qTx] * (4.5 / 6)) / 3.0

    def sampleOnPoint(self, point):
        # We only check the end of the road segment.
        pX, pY = self.getTextureCords(point)

        return self.map[pY][pX]


def rotatePoint2D(point, angle):
    theta = (angle / 360) * 2 * mt.pi
    cosTheta = mt.cos(theta)
    sinTheta = mt.sin(theta)

    newX = point.x * cosTheta - point.y * sinTheta
    newY = point.x * sinTheta + point.y * cosTheta

    return ExtendedPoint(newX, newY)


def segmentWithinLimit(segment):
    pOk = (0 < segment.p.x < 1) and (0 < segment.p.y < 1)
    qOk = (0 < segment.q.x < 1) and (0 < segment.q.y < 1)

    return pOk and qOk


def makeInitialSegments():
    # Here we set a segments to start in the middle at (1/2, 1/2)

    root = Segment(ExtendedPoint(0.5, 0.5), ExtendedPoint(0.5, 0.5 + config["highwayLength"]), True)
    # make the points point the roadSegment object that they create

    if config["twoStartSegment"]:

        secondRoot = Segment(ExtendedPoint(0.5, 0.5), ExtendedPoint(0.5, 0.5 - config["highwayLength"]), True)

        return [Item(root, 0.0), Item(secondRoot, 0.0)]
    else:
        return [Item(root, 0.0)]


def applyLocalConstraints(minSegment, segments, legalAreaMap):
    legalArea = segmentWithinLimit(minSegment) and (legalAreaMap.sampleOnPoint(minSegment.q) > 0.4)
    # Alter segment somewhat if it doesn't fit

    if legalArea:
        # The area is legal now let check if we can connect the segment

        closeSegments = segments.getCloseSegments(minSegment)

        if len(closeSegments) == 1:
            return True

        segmentLength = config["highwayLength" if minSegment.highway else "normalLength"]

        bestPoint = minSegment.q
        bestPointDistance = mt.inf

        prioValue = 0

        for midPoint in closeSegments:

            closeSegment = midPoint.data
            # The segment is intersection existing road
            newPoint, tCloseSegment, tMinSegment = closeSegment.intersection(minSegment)

            if newPoint:
                if newPoint != minSegment.p:  # It's okey if to segments share first node.

                    distanceNewPoint = minSegment.p.distanceTo(newPoint)

                    if distanceNewPoint < bestPointDistance or prioValue != 3:

                        if tCloseSegment < 0.35:
                            newPoint = closeSegment.p

                        elif tCloseSegment > 0.65:
                            newPoint = closeSegment.q

                        bestPointDistance = distanceNewPoint
                        bestPoint = newPoint

                    prioValue = 3

            if prioValue <= 2:
                pDistance = closeSegment.p.distanceTo(minSegment.q)
                qDistance = closeSegment.q.distanceTo(minSegment.q)

                if pDistance <= segmentLength * config["connectCrossingThreshold"]:
                    if prioValue == 2:
                        if pDistance < bestPointDistance:
                            bestPointDistance = pDistance
                            bestPoint = closeSegment.p
                    else:
                        bestPointDistance = pDistance
                        bestPoint = closeSegment.p
                        prioValue = 2

                if qDistance <= segmentLength * config["connectCrossingThreshold"]:
                    if prioValue == 2:
                        if qDistance < bestPointDistance:
                            bestPointDistance = qDistance
                            bestPoint = closeSegment.q
                    else:
                        bestPointDistance = qDistance
                        bestPoint = closeSegment.q
                        prioValue = 2

                # Check if we can extend the road.

                if prioValue <= 1:
                    dist, newPoint, tValue = closeSegment.getClosestPointOnSegment(minSegment.q)
                    if dist < segmentLength * config["okToExtendFactor"]:
                        if newPoint != minSegment.p:
                            if prioValue == 1:
                                if dist < bestPointDistance:
                                    bestPointDistance = dist
                                    bestPoint = newPoint
                            else:
                                bestPointDistance = dist
                                bestPoint = newPoint
                                prioValue = 1

        if minSegment.p.distanceTo(bestPoint) > (config["normalLength"] * config["streetMinLength"]):

            minSegment.q = bestPoint

            minSegmentDir = minSegment.getDirectionVector().setLength(1)
            nStreetConnectionsP = 0
            nStreetConnectionsQ = 0
            sameDirectionValue = 2.0

            minDist = segmentLength * 0.2

            for midPoint in closeSegments:

                closeSegment = midPoint.data

                # The minSegment will always be connected to its parent. This is okay but we need to check this.
                # It could be the case that the start point have generated a new crossing or have connected to a
                # crossing. Then we shouldn't accept this segment.
                dist, correspondingLinePoint, tValue = closeSegment.getClosestPointOnSegment(minSegment.p)
                if dist <= minDist:

                    nStreetConnectionsP += 1

                    closeSegmentDir = closeSegment.getDirectionVector().setLength(1)

                    angleForward = mt.acos(clamp(closeSegmentDir.dotProduct(minSegmentDir), -0.999, 0.999))
                    # THIS IS IN RADIANS
                    angleBackward = mt.pi - angleForward

                    forwardCondition = (angleForward > config["sameDirectionThreshold"] or tValue > 0.999)
                    backwardCondition = (angleBackward > config["sameDirectionThreshold"] or tValue < 0.001)

                    # tValue > 0.5. The starting point of the minSegment is very close to an end with another street

                    if (not (forwardCondition and backwardCondition)) or nStreetConnectionsP >= config[
                        "okNumberOfStreetConnections"]:
                        return False

                dist, correspondingLinePoint, tValue = closeSegment.getClosestPointOnSegment(minSegment.q)

                if dist <= minDist:

                    nStreetConnectionsQ += 1

                    closeSegmentDir = closeSegment.getDirectionVector().setLength(1)

                    angleForward = mt.acos(clamp(closeSegmentDir.dotProduct(minSegmentDir), -0.999, 0.999))
                    # THIS IS IN RADIANS
                    angleBackward = mt.pi - angleForward

                    forwardCondition = (angleForward > config["sameDirectionThreshold"] or tValue < 0.001)
                    backwardCondition = (angleBackward > config["sameDirectionThreshold"] or tValue > 0.999)

                    # tValue > 0.5. The starting point of the minSegment is very close to an end with another street

                    if (not (forwardCondition and backwardCondition)) or nStreetConnectionsQ >= config[
                        "okNumberOfStreetConnections"]:
                        return False

            return True

    return False


def findSuitableNewHighwaySegment(startPoint, dirVector, pMap):
    n = config["numberOfNewHighwayChecking"]
    angleStep = 2 * config["highwayMaxAngleTurn"] / n

    bestSegment = Segment(startPoint, startPoint.addVector(dirVector), True)
    bestPopulation = -mt.inf

    halfN = n // 2
    for i in range(-halfN, halfN + 1):
        angle = angleStep * i
        newDirection = rotatePoint2D(dirVector, angle)
        newSegment = Segment(startPoint, startPoint.addVector(newDirection), True)

        popScore = pMap.sampleWholeSegment(newSegment)
        if popScore > bestPopulation:
            bestSegment = newSegment
            bestPopulation = popScore

    return bestSegment


def globalGoalsGenerate(minItem, popMap):
    """
    When an area with no population is reached, the streets stop growing
    :param minItem:
    :param popMap:
    :return:
    """
    minSegment = minItem.data
    newSegments = []

    if popMap.sampleOnPoint(minSegment.q) <= (config["populationThreshold"] + (-0.15 if minSegment.highway else 0.15)):
        return newSegments

    directionVector = minSegment.getDirectionVector().setLength(
        config["highwayLength"] if minSegment.highway else config["normalLength"])

    branchResult = rd.random()
    dirResult = rd.random()
    angleResult = (90 + (rd.uniform(-1, 1) * config["branchingNoise"])) * (-1 if rd.random() <= 0.5 else 1)
    if minSegment.highway:

        newSegments.append(findSuitableNewHighwaySegment(minSegment.q, directionVector, popMap))

        if branchResult < config["highwayNewBranchProbability"]:
            newSegments.append(
                Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(directionVector, angleResult)), True))

        else:
            if branchResult < config["normalNewBranchProbability"]:
                normalStreetDir = directionVector.setLength(config["normalLength"])

                newSegments.append(
                    Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(normalStreetDir, angleResult)), False))

    else:
        # The street is normal we don't need to sample the surrounding
        newSegments.append(Segment(minSegment.q, minSegment.q.addVector(directionVector)))
        if branchResult < config["normalNewBranchProbability"]:
            newSegments.append(
                Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(directionVector, angleResult)), False))

    result = [Item(s, minItem.prioValue + (config["highwayDelayFactor"] if s.highway else config["normalDelayFactor"]))
              for s in newSegments]
    result[0].prioValue -= 0.01
    return result


def generateNetwork():
    segments = SegmentsContainer()
    popMap = Texture("textures/noise/simplex.png")
    waterMap = Texture("")
    Q = PriorityQueue()
    Q.pushAll(makeInitialSegments())

    segmentTimeVector = []

    for i in range(config["numberOfSegments"]):

        segmentEvalStartTime = tm.time()

        if Q.empty():
            break

        minItem = Q.pop()
        minSegment = minItem.data
        accepted = applyLocalConstraints(minSegment, segments, waterMap)
        if accepted:
            segments.addSegment(minSegment)
            Q.pushAll(globalGoalsGenerate(minItem, popMap))

        segmentTimeVector.append(tm.time() - segmentEvalStartTime)

    return segments.allSegments, segmentTimeVector


if __name__ == '__main__':

    rd.seed(config["seed"])
    runningN = 25
    nTries = 150

    segments, timeVector = generateNetwork()
    t = np.array(timeVector)

    for i in tqdm(range(nTries)):

        segments, timeVector = generateNetwork()
        t = t+np.array(timeVector)

    t = t*(1/nTries)

    plt.plot(timeVector)
    plt.show()
