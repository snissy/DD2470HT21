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
from quads import Point
import functools

config = json.loads(open("config.json", mode="r").read())


class ExtendedPoint(qd.Point):

    def __init__(self, x, y, data=None):
        super().__init__(x, y, data)

    def distanceTo(self, other):
        return mt.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Segment:
    p: ExtendedPoint
    q: ExtendedPoint
    highway: bool

    def __init__(self, p, q, highway=False):

        self.p = p
        self.q = q
        self.highway = highway

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
        return heapq.heappop(self.__que).data

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
        self.map = np.array(textureName).mean(axis=2)
        self.dims = self.map.shape

    def getTextureCords(self, point):
        return mt.floor(point.x * self.dims[0]), mt.floor(point.y * self.dims[1])

    def sampleWholeSegment(self, segment):
        # TODO here it should be scaled accordingly. Not that this code assumes segment is inside (0, 1)

        pIndex = self.getTextureCords(segment.p)
        qIndex = self.getTextureCords(segment.q)
        halfIndex = ((pIndex[0] + qIndex[0]) // 2, (pIndex[1] + qIndex[1]) // 2)

        return (self.map[pIndex] + self.map[qIndex] + self.map[halfIndex]) / 3.0

    def sampleOnRoadEnd(self, segment):
        # We only check the end of the road segment.
        qX, qY = self.getTextureCords(segment.q)

        qX = (max(qX - 2, 0), min(qX + 2, self.dims[0] - 1))
        qY = (max(qY - 2, 0), min(qY + 2, self.dims[1] - 1))

        return np.mean(self.map[qX[0]:qX[1], qY[0]:qY[1]])


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
        
        I suggest that we we first try check if the segment is in a illegal area. 
    """
    # TODO implement legal Area checking
    # TODO use a water map.
    legalArea = segmentWithinLimit(minSegment)

    if legalArea:
        # The area is legal now let check if we can connect the segment
        endPoint = minSegment.q
        newPoint = None
        closeSegments = segments.getCloseSegments(endPoint)
        # Let's check if the segment intersect something
        for cs in closeSegments:
            csData = cs.data
            # The segment is intersection existing road
            newPoint = csData.intersection(minSegment)
            if newPoint:
                minSegment.q = newPoint
                return True

            # The end points is close to a crossing.
            pDistance = csData.p.distanceTo(endPoint)
            qDistance = csData.q.distanceTo(endPoint)

            if pDistance <= config["connectCrossingThreshold"]:
                minSegment.q = csData.p
                return True
            if qDistance <= config["connectCrossingThreshold"]:
                minSegment.q = csData.q
                return True

            # Check if we can extend the road.
            newQx = minSegment.p.x + (minSegment.q.x - minSegment.p.x) * config["extendFactorForNewCrossing"]
            newQy = minSegment.p.y + (minSegment.q.y - minSegment.p.y) * config["extendFactorForNewCrossing"]
            newQ = ExtendedPoint(newQx, newQy)

            newPoint = csData.intersection(Segment(minSegment.p, newQ))
            if newPoint:
                minSegment.q = newPoint
                return True

        return True

    # make checks if the area is okey
    # THe area is okay, let's check if we need to connect to anything.

    return False


def globalGoalsGenerate(minSegment, popMap):
    """
    When an area with no population is reached, the streets stop growing
    :param minSegment:
    :param popMap:
    :return:
    """
    # TODO this is next important thing to do.


    if popMap.populationOnRoadEnd(minSegment) <= config["populationThreshold"]:
        pass

    return [("Segment", 0)]


def generateNetwork():
    # road&query  ra = Road Attribute, qa = Query attribute, I guess?

    maxNSegments = 250
    segments = SegmentsContainer()
    popMap = Texture("textures/noise/simplex.png")
    waterMap = Texture("")
    Q = PriorityQueue()
    Q.pushAll(makeInitialSegments())

    while (not Q.empty()) and (len(segments) < maxNSegments):

        minSegment = Q.pop()
        accepted = applyLocalConstraints(minSegment, segments, popMap)
        if accepted:
            segments.addSegment(minSegment)

            # addZeroToThreeRoadsUsingGlobalGoals(Q, t+1, qa,ra)
            # We need to increment t+1, t will be minSegment.T
            Q.pushAll(globalGoalsGenerate(minSegment, popMap))

    # All segments have been created. Time render them.

    # pDel[] branch delay and deletion,
    # pRoadAttr[] for road data, e.g. length, angle, etc  --> ra In blogpost, i think
    # pRuleAttr[] for rule-specific attributes  --> qa In blogpost, i think
    #


def drawLine(p1, p2, c='black'):
    x_values = [p1.x, p2.x]
    y_values = [p1.y, p2.y]
    plt.plot(x_values, y_values, c)


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
    nHalfLength = nLength/2.0
    nfifthLength = nLength / 5.0
    nsixthLength = nLength / 6.0


    # Test 1
    s1 = Segment(ExtendedPoint(0.5, 0), ExtendedPoint(0.5, nLength))
    s2 = Segment(ExtendedPoint(0.5-nHalfLength, nHalfLength), ExtendedPoint(0.5+nHalfLength, nHalfLength))

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")
    plt.show()

    segments = SegmentsContainer()
    segments.addSegment(s2)

    applyLocalConstraints(s1, segments)

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")
    plt.show()

    # Test 2
    s1 = Segment(ExtendedPoint(0.5, 0), ExtendedPoint(0.5, nLength))

    s2 = Segment(ExtendedPoint(0.5 + nfifthLength, nLength+nfifthLength),
                 ExtendedPoint(0.5 + nfifthLength + nLength, nLength+nfifthLength))

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")
    plt.show()

    segments = SegmentsContainer()
    segments.addSegment(s2)

    applyLocalConstraints(s1, segments)

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")
    plt.show()

    # Test 3
    s1 = Segment(ExtendedPoint(0.5, 0), ExtendedPoint(0.5, nLength))
    s2 = Segment(ExtendedPoint(0.5 - nHalfLength, nLength + nsixthLength), ExtendedPoint(0.5 + nHalfLength, nLength + nsixthLength))

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")
    plt.show()

    segments = SegmentsContainer()
    segments.addSegment(s2)

    applyLocalConstraints(s1, segments)

    drawLine(s1.p, s1.q, "red")
    drawLine(s2.p, s2.q, "green")
    plt.show()


    pass


if __name__ == '__main__':
    # generateNetwork()
    # checkLineSegmentQuery()
    # Notes,
    # TODO If I have 4 days before the presentation is done I should try to create this code in c++
    # TODO add map size in config json that scales the range of the map

    testingLocalConstrains()
    """
    December 26/12
    I'm going to work with a normalized range of the map. All values has to been in the range (0, 1). 
    I've have noted that it is only the highways That follow the different roadPatterns. 

    """
    testDistanceFunction()
