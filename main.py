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

config = json.loads(open("config.json", mode="r").read())


class ExtendedPoint(qd.Point):

    def __init__(self, x, y, data=None):
        super().__init__(x, y, data)

    def distanceTo(self, other):
        return mt.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def addVector(self, other):
        newX = self.x + other.x
        newY = self.y + other.y

        return ExtendedPoint(newX, newY)

    def setLength(self, scale):
        """
        The vector will get the length of the scale
        :param scale:
        :return:
        """
        if abs(self.x - self.y) <= 0.00000001:
            print("WARNING: You get set length on zero vector")
            raise Exception("You get set length on zero Vector")

        mag = scale / ((self.x ** 2 + self.y ** 2) ** (1 / 2))
        newX = self.x * mag
        newY = self.y * mag
        return ExtendedPoint(newX, newY)

    def scale(self, s):
        self.x *= s
        self.y *= s

    def __eq__(self, other):
        return abs(self.x - other.x) <= 0.0000001 and abs(self.y - other.y) <= 0.0000001

    def dotProduct(self, other):
        return self.x * other.x + self.y * other.y


class Segment:
    p: ExtendedPoint
    q: ExtendedPoint
    highway: bool

    def __init__(self, p, q, highway=False):

        self.p = p
        self.q = q
        self.highway = highway

    def getDirectionVector(self):

        newPoint = ExtendedPoint(0, 0)
        newPoint.x = self.q.x - self.p.x
        newPoint.y = self.q.y - self.p.y

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
        # Have done some error checking this seems to work.
        # Note to myself. I am never going to write this code again.
        """
        a = self.p.x - other.p.x  # x1 - x3
        b = other.p.y - other.q.y  # y3 - y4
        c = self.p.y - other.p.y  # y1 - y3
        d = other.p.x - other.q.x  # x3 - x4
        e = self.p.x - self.q.x  # x1 - x2
        f = self.p.y - self.q.y  # y1 - y2

        determinant = (e * b - f * d)

        if -0.000001 <= determinant <= 0.000001:
            # The line parallel, this requires some more checking

            def is_between(p1, p3, p2):
                return - 0.000001 < (p1.distanceTo(p3) + p3.distanceTo(p2) - p1.distanceTo(p2)) < 0.000001

            if is_between(self.p, other.q, self.q):
                return other.q
            if is_between(self.p, other.p, self.q):
                return other.p

            return None

        else:
            t = (a * b - c * d) / determinant
            if 0 <= t <= 1:
                u = (a * f - c * e) / determinant

                if 0 <= u <= 1:
                    newX = self.p.x + (self.q.x - self.p.x) * t
                    newY = self.p.y + (self.q.y - self.p.y) * t
                    return ExtendedPoint(newX, newY)

        return None

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
        u = max(0, min(1, ((x3 - x1) * px + (y3 - y1) * py) / float(norm)))

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

    def getCloseSegments(self, point):
        # ((5)^(1/2)/2) = 1.1180 <= 1.12
        sLength = 1.5 * config["highwayLength"]

        minX = point.x - sLength
        maxX = point.x + sLength
        minY = point.y - sLength
        maxY = point.y + sLength
        pointBB = qd.BoundingBox(minX, minY, maxX, maxY)

        res = self.tree.within_bb(pointBB)
        return res


class Texture:

    def __init__(self, textureName):
        self.map = np.array(Image.open(textureName)).mean(axis=2)
        self.dims = self.map.shape

    def getTextureCords(self, point):
        x = mt.floor(point.x * self.dims[0])
        y = mt.floor(point.y * self.dims[1])
        return min(max(x, 0), self.dims[0] - 1), min(max(y, 0), self.dims[1] - 1)

    def sampleWholeSegment(self, segment):
        # TODO in the original paper inverseDistance scale the values in favor for the end of the segment

        pIndex = self.getTextureCords(segment.p)
        qIndex = self.getTextureCords(segment.q)
        halfIndex = ((pIndex[0] + qIndex[0]) // 2, (pIndex[1] + qIndex[1]) // 2)

        return (self.map[pIndex] * (1 / 6) + self.map[halfIndex] * (2 / 6) + self.map[qIndex] * (3 / 6)) / 3.0

    def sampleOnRoadEnd(self, segment):
        # We only check the end of the road segment.
        qX, qY = self.getTextureCords(segment.q)

        qX = (max(qX - 2, 0), min(qX + 2, self.dims[0] - 1))
        qY = (max(qY - 2, 0), min(qY + 2, self.dims[1] - 1))

        return np.mean(self.map[qX[0]:qX[1], qY[0]:qY[1]])


def rotatePoint2D(point, angle):
    # TODO cache these results of cos and sin calculations since they will be the same all the time.

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

    root = Segment(ExtendedPoint(0.5, 0.5), ExtendedPoint(0.5 + config["highwayLength"], 0.5), True)
    # make the points point the roadSegment object that they create

    if config["twoStartSegment"]:

        secondRoot = Segment(ExtendedPoint(0.5, 0.5), ExtendedPoint(0.5 - config["highwayLength"], 0.5), True)

        return [Item(root, 0.0), Item(secondRoot, 0.0)]
    else:
        return [Item(root, 0.0)]


def applyLocalConstraints(minSegment, segments):
    # + a quadtree in applyLocalConstraints
    """
    The localConstraints function executes the two following steps:

        • check if the road segment ends inside or crosses an illegal area.

        • search for intersection with other roads or for roads and crossings that are within a specified distance to the segment end.

        If the first check determines that the road end is inside water, a park or another illegal area,
        the system tries to readjust the values for the road segment in the following ways.

        • Prune the segment length up to a certain factor so that it fits inside the legal area of the starting point.
        • Rotate the segment within a maximal angle until it is completely inside the legal area. This allows the creation of roads that follow a coastline or a park boundary.
        • Highways are allowed to cross illegal area up to a specified length.

        The generated highway segment is flagged. At the geometry creation stage it can then be replaced by e.g. a bridge, or two tunnel entrances on both sides.

        Once all road ends are checked for being inside legal territory,
        the system now scans the surrounding of the road ends for other road to form crossings and intersections.


        If the localConstraints function finds a street within the given radius of the end of a segment it can modify
        the parameters for the following events illustrated in figure 8:

        • two streets intersect → generate a crossing.
        • ends close to an existing crossing → extend street, to reach the crossing.
        • close to intersecting → extend street to form intersection
    :param minSegment:
    :param segments:
    :param popMap: 
    :return:
    """""
    """
     The localConstraints function executes the two following steps:
        • check if the road segment ends inside or crosses an illegal area.
        • search for intersection with other roads or for roads and crossings that are within a specified distance to the segment end.
    """
    # TODO implement legal Area checking
    # TODO use a water map.
    legalArea = segmentWithinLimit(minSegment)

    if legalArea:
        # The area is legal now let check if we can connect the segment
        endPoint = minSegment.q
        bestPoint = minSegment.q
        bestPointDistance = mt.inf

        prioValue = 0
        newPoint = None
        closeSegments = segments.getCloseSegments(endPoint)
        # Let's check if the segment intersect something

        for midPoint in closeSegments:
            closeSegment = midPoint.data

            # The minSegment will always be connected to its parent. This is okay but we need to check this.
            # It could be the case that the start point have generated a new crossing or have connected to a
            # crossing. Then we shouldn't accept this segment.
            dist, correspondingLinePoint, tValue = closeSegment.getClosestPointOnSegment(minSegment.p)

            if dist <= 0.0001:

                d1 = closeSegment.getDirectionVector()
                d2 = minSegment.getDirectionVector()
                # TODO code repetition.
                if tValue < 0.001:
                    # TODO Config Value
                    sameDirectionValue = 1 - d1.dotProduct(d2)

                elif abs(tValue - 1) < 0.001:
                    d1.scale(-1)
                    sameDirectionValue = 1 - d1.dotProduct(d2)

                else:
                    return False
                    sameDirectionValue = 1 - abs(d1.dotProduct(d2))

                if sameDirectionValue < 0.15:
                    # TODO CONFIG VALUE SHOULD BE HERE.
                    return False

            # The segment is intersection existing road
            newPoint = closeSegment.intersection(minSegment)
            if newPoint:
                # TODO A line could intersect multiple lines. Then we need to pick the first/the closest intersection
                if newPoint != minSegment.p:  # It's okey if to segments share first node.

                    distanceNewPoint = minSegment.p.distanceTo(newPoint)
                    if prioValue == 3:
                        if distanceNewPoint < bestPointDistance:
                            # TODO remove code repetition
                            bestPointDistance = distanceNewPoint
                            bestPoint = newPoint
                    else:
                        bestPointDistance = distanceNewPoint
                        bestPoint = newPoint
                        prioValue = 3

            # The end points is close to a crossing.

            if prioValue < 3:

                # TODO ADD limit crossing limit size.

                pDistance = closeSegment.p.distanceTo(endPoint)
                qDistance = closeSegment.q.distanceTo(endPoint)
                if pDistance <= config["connectCrossingThreshold"]:
                    if prioValue == 2:
                        if pDistance < bestPointDistance:
                            bestPointDistance = pDistance
                            bestPoint = closeSegment.p
                    else:
                        bestPointDistance = pDistance
                        bestPoint = closeSegment.p
                        prioValue = 2

                if qDistance <= config["connectCrossingThreshold"]:
                    if prioValue == 2:
                        if qDistance < bestPointDistance:
                            bestPointDistance = qDistance
                            bestPoint = closeSegment.q
                    else:
                        bestPointDistance = qDistance
                        bestPoint = closeSegment.q
                        prioValue = 2

                # Check if we can extend the road.

                if prioValue < 1:
                    dist, newPoint, tValue = closeSegment.getClosestPointOnSegment(endPoint)
                    if dist < config["okToExtendFactor"]:
                        if newPoint != minSegment.p:
                            if prioValue == 1:
                                if dist < bestPointDistance:
                                    bestPointDistance = dist
                                    bestPoint = newPoint
                            else:
                                bestPointDistance = dist
                                bestPoint = newPoint
                                prioValue = 1

        if minSegment.p.distanceTo(bestPoint) > config["normalLength"]*0.45:
            minSegment.q = bestPoint
            return True

    # make checks if the area is okey
    # THe area is okay, let's check if we need to connect to anything.

    return False


def globalGoalsGenerate(minItem, popMap):
    """
    When an area with no population is reached, the streets stop growing
    :param minItem:
    :param popMap:
    :return:
    """
    # TODO we need to add some noise on the branching.
    # FIX SOME OF THE BRANCHING LIKE ONLY BRANCH IN ONE DIRECTION either left or right.

    minSegment = minItem.data
    newSegments = []

    if popMap.sampleOnRoadEnd(minSegment) <= config["populationThreshold"] + (0.15 if minSegment.highway else 0):
        return newSegments

    # This is for the new Segment going in the forward direction.

    directionVector = minSegment.getDirectionVector()
    normalStreetDir = directionVector.setLength(config["normalLength"])

    # IT ONLY BRANCH AT ONE DIRECTION
    leftResult = rd.random()
    rightResult = rd.random()

    if minSegment.highway:
        # We should sample the surrounding area for the best highway Forward branch
        # First make the forward segment.
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

        # The branching segment with angle 90
        if leftResult < config["highwayNewBranchProbability"]:
            newSegments.append(Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(directionVector, 90)), True))
        else:
            if leftResult < config["normalNewBranchProbability"]:
                newSegments.append(
                    Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(normalStreetDir, 90)), False))

        if rightResult < config["highwayNewBranchProbability"]:
            newSegments.append(Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(directionVector, -90)), True))
        else:
            if rightResult < config["normalNewBranchProbability"]:
                newSegments.append(
                    Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(normalStreetDir, -90)), False))

    else:
        # The street is normal we don't need to sample the surrounding
        newSegments.append(Segment(minSegment.q, minSegment.q.addVector(directionVector)))

        if leftResult < config["normalNewBranchProbability"]:
            newSegments.append(Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(directionVector, 90)), False))

        if rightResult < config["normalNewBranchProbability"]:
            newSegments.append(
                Segment(minSegment.q, minSegment.q.addVector(rotatePoint2D(directionVector, -90)), False))

    #TODO this is ugly
    result = [Item(s, minItem.prioValue + (config["highwayDelayFactor"] if s.highway else config["normalDelayFactor"]))
              for
              s in newSegments]

    result[0].prioValue -= 2
    return result


def generateNetwork():
    # road&query  ra = Road Attribute, qa = Query attribute, I guess?

    maxNSegments = 250
    segments = SegmentsContainer()
    popMap = Texture("textures/noise/simplex.png")
    waterMap = Texture("textures/water/w1.png")
    Q = PriorityQueue()
    Q.pushAll(makeInitialSegments())

    #plt.xlim([0.0, 1 * popMap.dims[0]])
    #plt.ylim([0.0, 1 * popMap.dims[1]])

    # while (not Q.empty()) and (len(segments) < maxNSegments):
    n = 2500
    for i in tqdm(range(n)):
        if Q.empty():
            n = i
            break
        # if i == 208:
        #     print("let's check something")
        minItem = Q.pop()
        minSegment = minItem.data
        accepted = applyLocalConstraints(minSegment, segments)
        if accepted:
            segments.addSegment(minSegment)
            # addZeroToThreeRoadsUsingGlobalGoals(Q, t+1, qa,ra)
            # We need to increment t+1, t will be minSegment.T
            drawLine(minSegment.p, minSegment.q, popMap.dims, "#00ff00" if minSegment.highway else "#ff0000")

            Q.pushAll(globalGoalsGenerate(minItem, popMap))
        #else:
            #drawLine(minSegment.p, minSegment.q, popMap.dims, "#83a653" if minSegment.highway else "#a65353")

        # All segments have been created. Time render them.
        # drawSegments(segments, i)
        # for s in Q.getAllData():drawLine(s.p, s.q, "orange")
        plt.title("Iteration: {}".format(i), fontsize=12)
        #plt.savefig("outputImg/roadGeneration-iter-{}".format(i))

    plt.imshow(popMap.map, cmap="gray")
    plt.show()

    makeGif(n)
    # pDel[] branch delay and deletion,
    # pRoadAttr[] for road data, e.g. length, angle, etc  --> ra In blogpost, i think
    # pRuleAttr[] for rule-specific attributes  --> qa In blogpost, i think
    #


def drawLine(p1, p2, dimsFactors, c='black', order=10):
    x_values = [p1.x * dimsFactors[0], p2.x * dimsFactors[0]]
    y_values = [p1.y * dimsFactors[1], p2.y * dimsFactors[1]]
    plt.plot(x_values, y_values, c, marker='o', zorder=order, markersize=2.5)


def drawSegments(segments, i):
    plt.xlim([0.35, 0.65])
    plt.ylim([0.35, 0.65])
    for s in segments.allSegments:
        drawLine(s.p, s.q, "green" if s.highway else "red")
    plt.title("Iteration: {}".format(i), fontsize=12)
    plt.savefig("outputImg/roadGeneration-iter-{}".format(i))
    plt.clf()


def makeGif(iterations):
    fileNames = ["outputImg/roadGeneration-iter-{}.png".format(i) for i in range(iterations)]
    with imageio.get_writer('outputImg/Final.gif', mode='I') as writer:
        for filename in tqdm(fileNames):
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
        print("Number of close segments {}".format(len(closeSegments)))
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


if __name__ == '__main__':

    # for i in range(100):
    #     print(i)
    #     rd.seed(i)
    #     generateNetwork()

    rd.seed(1)
    generateNetwork()


    # checkLineSegmentQuery()
    # TODO If I have 4 days before the presentation is done I should try to create this code in c++
    # TODO add map size in config json that scales the range of the map
    """
    December 26/12
    I'm going to work with a normalized range of the map. All values has to been in the range (0, 1). 
    I've have noted that it is only the highways That follow the different roadPatterns. 
    """
