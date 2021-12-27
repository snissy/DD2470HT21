from dataclasses import dataclass, field
import heapq
import math as mt
from typing import Any
import json
from PIL import Image
import numpy as np
import quads as qd

config = json.loads(open("config.json", mode="r").read())


@dataclass
class Segment:
    p: qd.Point
    q: qd.Point
    highway: bool

    def intersection(self, other):

        """
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        self.p = (x1, y1)
        self.q = (x2, y2)
        other.p = (x3, y3)
        other.q = (x4, y4)
        :param other:
        :return:
        # TODO check THAT THIS IS CORRECT VERY DANGEROUS DO BUGS
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
            def distance(p1, p2):
                return mt.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

            def is_between(p1, p3, p2):
                return - 0.000001 < (distance(p1, p3) + distance(p3, p2) - distance(p1, p2)) < 0.000001

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
                    return qd.Point(newX, newY)

        return None


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
        self.tree = qd.QuadTree((0, 0), 1, 1)

    def addSegment(self, s):
        self.allSegments.append(s)
        self.tree.insert(s.p)
        self.tree.insert(s.q)

    def __len__(self):
        return len(self.allSegments)

    def getCloseSegments(self, segment):
        """
        bb = quads.BoundingBox(min_x=-1, min_y=-2, max_x=2, max_y=2)
        tree.within_bb(bb)
        :param segment:
        :return:
        """
        sLength = config["highwayLength" if segment.highway else "normalLength"]

        minX = segment.q.x - sLength
        maxX = segment.q.x + sLength
        minY = segment.q.Y - sLength
        maxY = segment.q.Y + sLength
        pointBB = qd.BoundingBox(minX, minY, maxX, maxY)
        return self.tree.within_bb(pointBB)


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
    # TODO we don't have to check both points, this is made with safety within mind.
    # THis can be removed later.

    pOk = (0 < segment.p.x < 1) and (0 < segment.p.y < 1)
    qOk = (0 < segment.q.x < 1) and (0 < segment.q.y < 1)

    return pOk and qOk


def makeInitialSegments():
    """

      // setup first segments in queue
  const rootSegment = new Segment(
    { x: 0, y: 0 },
    { x: config.HIGHWAY_SEGMENT_LENGTH, y: 0 },
    0,
    { highway: !config.START_WI1qssz1§TH_NORMAL_STREETS },
  );
  if (!config.TWO_SEGMENTS_INITIALLY) return [rootSegment];
  const oppositeDirection = rootSegment.clone();
  const newEnd = {
    x: rootSegment.start.x - config.HIGHWAY_SEGMENT_LENGTH,
    y: oppositeDirection.end.y,
  };
  oppositeDirection.end = newEnd;
  oppositeDirection.links.b.push(rootSegment);
  rootSegment.links.b.push(oppositeDirection);
  return [rootSegment, oppositeDirection];
    :return:

    """
    # TODO note that you can store data in the pointObject which is good.
    # Here we set a segments to start in the middle at (1/2, 1/2)

    root = Segment(qd.Point(0.5, 0.5), qd.Point(0.5 + config["highwayLength"], 0.5), True)

    # make the points point the roadSegment object that they create

    if config["twoStartSegment"]:

        secondRoot = Segment(root.p, qd.Point(0.5 - config["highwayLength"], 0.5), True)

        return [Item(root, 0.0), Item(secondRoot, 0.0)]
    else:
        return [Item(root, 0.0)]


def applyLocalConstraints(minSegment, segments, popMap):
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
    # TODO THis is the next important Step
    """
     The localConstraints function executes the two following steps:
        • check if the road segment ends inside or crosses an illegal area.
        • search for intersection with other roads or for roads and crossings that are within a specified distance to the segment end.
        
        I suggest that we we first try check if the segment is in a illegal area. 
    """

    illegalArea = True

    # TODO use a water map.
    pass

    # make checks if the area is okey

    # THe area is okay, let's check if we need to connect to anything.

    return False


def globalGoalsGenerate(minSegment, popMap):
    """
    When an area with no population is reached, the streets stop growing

    Roadpatt
    :param minSegment:
    :param popMap:
    :return:
    """
    if popMap.populationOnRoadEnd(minSegment) <= config["populationThreshold"]:
        pass

    return [("Segment", 0)]


def generateNetwork():
    # road&query  ra = Road Attribute, qa = Query attribute, I guess?

    maxNSegments = 200
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


if __name__ == '__main__':
    # generateNetwork()

    # generateNetwork()
    pass

    # Notes,
    # TODO If I have 4 days before the presentation is done I should try to create this code in c++
    # TODO add map size in config json that scales the range of the map
    """
    December 26/12
    I'm going to work with a normalized range of the map. All values has to been in the range (0, 1). 
    I've have noted that it is only the highways That follow the different roadPatterns. 

    
    """
