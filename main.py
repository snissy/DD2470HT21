from dataclasses import dataclass, field
import heapq
import math as mt
from typing import Any
import json
from PIL import Image
import numpy as np
import quads as qd
import random as rd
from matplotlib import pyplot as plt
import imageio
from tqdm import tqdm
import cProfile as profile
import time as tm
import cv2

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


def drawApplyLocalCase(minSegment, closeSegments):
    plt.clf()
    drawLine(minSegment.p, minSegment.q, (1, 1), "blue")
    m = minSegment.getMidPoint()
    plt.plot(m.x, m.y, "blue", marker='o', markersize=3)
    for midPoint in closeSegments:
        closeSegment = midPoint.data
        color = "#448700" if closeSegment.highway else "#ffaa00"
        drawLine(closeSegment.p, closeSegment.q, (1, 1), color)
        m = closeSegment.getMidPoint()
        plt.plot(m.x, m.y, color, marker='o', markersize=3)

        dirPoint = closeSegment.getDirectionVector().setLength(0.005)
        dirPoint.x += closeSegment.p.x
        dirPoint.y += closeSegment.p.y
        plt.plot(dirPoint.x, dirPoint.y, "red", marker='o', markersize=3, zorder=12)
    plt.show()


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

    if config["onlyHighway"] and not minSegment.highway:
        #TODO REMOVE BEFORE BENCH
        return False

    if legalArea:
        # The area is legal now let check if we can connect the segment

        closeSegments = segments.getCloseSegments(minSegment)

        if len(closeSegments) == 1:
            return True

        segmentLength = config["highwayLength" if minSegment.highway else "normalLength"]

        bestPoint = minSegment.q
        bestPointDistance = mt.inf

        prioValue = 0

        # Let's check if the segment intersect something
        if config["iteration"] == config["iterationStop"]:
            # TODO remove Before BENCH

            drawApplyLocalCase(minSegment, closeSegments)
            print(config["iteration"])

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

                # TODO ADD limit crossing limit size.
                # The end points is close to a crossing.
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

            if config["iteration"] == config["iterationStop"]:
                # TODO remove Before BENCH
                drawApplyLocalCase(minSegment, closeSegments)

            minSegmentDir = minSegment.getDirectionVector().setLength(1)
            nStreetConnectionsP = 0
            nStreetConnectionsQ = 0
            sameDirectionValue = 2.0

            minDist = segmentLength * 0.2
            ## TODO CONFIG VALUE. You could experiment with this

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


def globalGoalsGenerate(minItem, popMap):
    """
    When an area with no population is reached, the streets stop growing
    :param minItem:
    :param popMap:
    :return:
    """
    minSegment = minItem.data
    newSegments = []

    # TODO add config value

    if popMap.sampleOnPoint(minSegment.q) <= (config["populationThreshold"] + (-0.15 if minSegment.highway else 0.15)):

        return newSegments

    directionVector = minSegment.getDirectionVector().setLength(
        config["highwayLength"] if minSegment.highway else config["normalLength"])

    branchResult = rd.random()
    dirResult = rd.random()
    angleResult = (90 + (rd.uniform(-1, 1) * config["branchingNoise"])) * (-1 if rd.random() <= 0.5 else 1)
    if minSegment.highway:

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
    # road&query  ra = Road Attribute, qa = Query attribute, I guess?

    maxNSegments = config["numberOfSegments"]
    segments = SegmentsContainer()
    popMap = Texture("textures/noise/simplex.png")
    #waterMap = Texture("textures/water/stockHolmMask.png")
    waterMap = Texture("")
    Q = PriorityQueue()

    startTime = tm.time()
    Q.pushAll(makeInitialSegments())

    # plt.xlim([248, 272])
    # plt.ylim([72, 89])

    # while (not Q.empty()) and (len(segments) < maxNSegments):
    n = maxNSegments
    for i in tqdm(range(n)):

        config["iteration"] = i

        if Q.empty():
            n = i
            break

        minItem = Q.pop()
        minSegment = minItem.data
        accepted = applyLocalConstraints(minSegment, segments, waterMap)
        if accepted:
            segments.addSegment(minSegment)
            # addZeroToThreeRoadsUsingGlobalGoals(Q, t+1, qa,ra)
            # We need to increment t+1, t will be minSegment.T

            if config["gifModeOn"]:
                args = ("#087800", 5) if minSegment.highway else ("#ffaa00", 1)
                drawLine(minSegment.p, minSegment.q, (popMap.dimX, popMap.dimY), *args)

            Q.pushAll(globalGoalsGenerate(minItem, popMap))
        #
        else:
            if config["drawNoneAcceptableSegments"]:
                if config["onlyHighway"]:
                    if minSegment.highway:
                        args = ("#087800", 5)
                        drawLine(minSegment.p, minSegment.q, (popMap.dimX, popMap.dimY), *args)
                else:
                    args = ("#707d40", 5) if minSegment.highway else ("#ff4d00", 1)
                    drawLine(minSegment.p, minSegment.q, (popMap.dimX, popMap.dimY), *args)

        # All segments have been created. Time render them.
        # drawSegments(segments, i)
        # for s in Q.getAllData():drawLine(s.p, s.q, "orange")

        if config["gifModeOn"]:
            plt.title("Iteration: {}".format(i), fontsize=12)
            plt.savefig("outputImg/roadGeneration-iter-{}".format(i))

        # if i == 30:
        #     plt.show()

    print("The system generated the network on {} seconds".format((round(tm.time() - startTime, 3))))

    if not config["gifModeOn"]:
        for s in segments.allSegments:
            drawLine(s.p, s.q, (popMap.dimX, popMap.dimY), "#087800" if s.highway else "#ffaa00")

    # plt.imshow(cv2.resize(waterMap.map, popMap.map.shape), cmap="gray")
    plt.imshow(np.multiply(popMap.map, cv2.resize(waterMap.map, popMap.map.shape)), cmap="gray")

    plt.show()

    if config["gifModeOn"]:
        makeGif(n)
    # pDel[] branch delay and deletion,
    # pRoadAttr[] for road data, e.g. length, angle, etc  --> ra In blogpost, i think
    # pRuleAttr[] for rule-specific attributes  --> qa In blogpost, i think
    #


def drawLine(p1, p2, dimsFactors, c='black', order=10, width=1.5):
    x_values = [p1.x * dimsFactors[0], p2.x * dimsFactors[0]]
    y_values = [p1.y * dimsFactors[1], p2.y * dimsFactors[1]]
    plt.plot(x_values, y_values, c, marker='o', zorder=order, markersize=config["markerSize"], linewidth=width)


def drawSegments(segments, i):
    plt.xlim([0.35, 0.65])
    plt.ylim([0.35, 0.65])
    for s in segments.allSegments:
        drawLine(s.p, s.q, "green" if s.highway else "red")
    plt.title("Iteration: {}".format(i), fontsize=12)
    plt.savefig("outputImg/roadGeneration-iter-{}".format(i))
    plt.clf()


def makeGif(iterations):
    print("Starting to create the GIF")
    fileNames = ["outputImg/roadGeneration-iter-{}.png".format(i) for i in range(iterations)]
    with imageio.get_writer('outputImg/Final.gif', mode='I') as writer:
        for count, filename in enumerate(tqdm(fileNames)):
            if count % config["gifSpeed"] == 0:
                image = imageio.imread(filename)
                writer.append_data(image)


def checkLineSegmentQuery():
    for ii in range(5):
        # generateNetwork()
        nSegments = 5000
        segments = SegmentsContainer()
        p = ExtendedPoint(rd.random(), rd.random())
        q = ExtendedPoint(rd.random(), rd.random())
        qp = ExtendedPoint(q.x - p.x, q.y - p.y)
        norm = (1 / (qp.x ** 2 + qp.y ** 2) ** (1 / 2)) * config["normalLength"]
        q = ExtendedPoint(p.x + qp.x * norm, p.y + qp.y * norm)

        startPoint = p
        checkPoint = q
        checkSegment = Segment(startPoint, checkPoint)

        drawLine(startPoint, checkPoint, "blue")
        plt.plot(checkPoint.x, checkPoint.y, 'ro', markersize=1)

        for i in range(nSegments):
            p = ExtendedPoint(rd.random(), rd.random())
            q = ExtendedPoint(rd.random(), rd.random())
            qp = ExtendedPoint(q.x - p.x, q.y - p.y)
            norm = (1 / (qp.x ** 2 + qp.y ** 2) ** (1 / 2)) * config["normalLength"]
            q = ExtendedPoint(p.x + qp.x * norm, p.y + qp.y * norm)
            newSegment = Segment(p, q)
            segments.addSegment(newSegment)

        closeSegments = segments.getCloseSegments(checkPoint)

        for s in segments.allSegments:

            ok = False
            for closeS in closeSegments:

                if closeS.data == s:

                    if checkSegment.intersection(closeS.data):

                        drawLine(s.p, s.q, "r")

                    else:
                        drawLine(s.p, s.q, "y")

                    ok = True

            if not ok:
                drawLine(s.p, s.q, "g")

        plt.show()


def testDistanceFunction():
    p1 = ExtendedPoint(0, 0)
    p2 = ExtendedPoint(1, 1)

    print(p1.distanceTo(p2))


def testingLocalConstrains():
    nLength = config["normalLength"]
    nHalfLength = nLength / 2.0
    nfifthLength = nLength / 5.0
    nsixthLength = nLength / 6.0

    # Test 1
    s1 = Segment(ExtendedPoint(0.5, 0.0001), ExtendedPoint(0.5, nLength))
    s2 = Segment(ExtendedPoint(0.5 - nHalfLength, nHalfLength), ExtendedPoint(0.5 + nHalfLength, nHalfLength))

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")

    segments = SegmentsContainer()
    segments.addSegment(s2)

    applyLocalConstraints(s1, segments)

    drawLine(s1.p, s1.q, "b")
    plt.show()

    # Test 2
    s1 = Segment(ExtendedPoint(0.5, 0.0001), ExtendedPoint(0.5, nLength))

    s2 = Segment(ExtendedPoint(0.5 + nfifthLength, nLength + nfifthLength),
                 ExtendedPoint(0.5 + nfifthLength + nLength, nLength + nfifthLength))

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")

    segments = SegmentsContainer()
    segments.addSegment(s2)

    applyLocalConstraints(s1, segments)

    drawLine(s1.p, s1.q, "b", 0)
    plt.show()

    # Test 3
    s1 = Segment(ExtendedPoint(0.5, 0.0001), ExtendedPoint(0.5, nLength))
    s2 = Segment(ExtendedPoint(0.5 - nHalfLength, nLength + nsixthLength),
                 ExtendedPoint(0.5 + nHalfLength, nLength + nsixthLength))

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")

    segments = SegmentsContainer()
    segments.addSegment(s2)

    applyLocalConstraints(s1, segments)

    drawLine(s1.p, s1.q, "b", 0)
    plt.show()

    pass


def testingLocalConstrains2():
    ## THIS IS a case where the function fails.

    config["iteration"] = 500000
    # config["iteration"] = config["iterationStop"]

    waterMap = Texture()

    NL = config["normalLength"]
    segments = SegmentsContainer()

    origin = ExtendedPoint(0.5, 0.5)
    S1 = Segment(origin, origin.addVector(ExtendedPoint(-NL, 0)))
    S2 = Segment(origin, origin.addVector(ExtendedPoint(0, NL)))
    S3 = Segment(origin, origin.addVector(ExtendedPoint(NL, 0)))
    S4 = Segment(origin, origin.addVector(ExtendedPoint(-0.002, -NL)))

    segments.addSegment(S1)
    segments.addSegment(S2)
    segments.addSegment(S3)
    segments.addSegment(S4)

    testSegment = Segment(origin.addVector(ExtendedPoint(0, -NL)), origin)

    print(applyLocalConstraints(testSegment, segments, waterMap))


def generateNetworkNoPlotWithProfiling():
    # road&query  ra = Road Attribute, qa = Query attribute, I guess?

    maxNSegments = config["numberOfSegments"]
    segments = SegmentsContainer()
    popMap = Texture("textures/noise/simplex.png")
    # waterMap = Texture("textures/water/stockHolmMask.png")
    waterMap = Texture("")
    Q = PriorityQueue()

    # In outer section of code
    pr = profile.Profile()
    pr.disable()
    pr.enable()

    Q.pushAll(makeInitialSegments())

    n = maxNSegments
    for i in range(n):

        if Q.empty():
            break

        minItem = Q.pop()
        minSegment = minItem.data
        accepted = applyLocalConstraints(minSegment, segments, waterMap)
        if accepted:
            segments.addSegment(minSegment)
            # addZeroToThreeRoadsUsingGlobalGoals(Q, t+1, qa,ra)
            # We need to increment t+1, t will be minSegment.T
            Q.pushAll(globalGoalsGenerate(minItem, popMap))

    pr.disable()
    pr.dump_stats('profile.pstat')

    return segments


if __name__ == '__main__':
    # for i in range(100):
    #     print(i)
    #     rd.seed(i)
    #     generateNetwork()

    rd.seed(config["seed"])
    #generateNetworkNoPlotWithProfiling()
    generateNetwork()



