from dataclasses import dataclass
import quads as qd
from matplotlib import pyplot as plt
import math as mt
import random as rd


def drawLine(p1, p2):
    x_values = [p1.x, p2.x]
    y_values = [p1.y, p2.y]
    plt.plot(x_values, y_values, 'black')


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
        """
        # TODO check THAT THIS IS CORRECT VERY DANGEROUS DO BUGS
        a = self.p.x - other.p.x  # x1 - x3
        b = other.p.y - other.q.y  # y3 - y4
        c = self.p.y - other.p.y  # y1 - y3
        d = other.p.x - other.q.x  # x3 - x4
        e = self.p.x - self.q.x  # x1 - x2
        f = self.p.y - self.q.y  # y1 - y2

        determinate = (e * b - f * d)

        if -0.000001 <= determinate <= 0.000001:
            # The line parallel we don't care if segments lie on each other producers in local constraint will handle that.
            def distance(p1, p2):
                return mt.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

            def is_between(p1, p3, p2):

                suggestedDist = distance(p1, p3) + distance(p3, p2)

                return - 0.000001 < (distance(p1, p3) + distance(p3, p2) - distance(p1, p2)) < 0.000001

            if is_between(self.p, other.q, self.q):
                return other.q
            if is_between(self.p, other.p, self.q):
                return other.p

            return None

        else:
            t = (a * b - c * d) / determinate
            if 0 <= t <= 1:
                u = (a * f - c * e) / determinate

                if 0 <= u <= 1:
                    newX = self.p.x + (self.q.x - self.p.x) * t
                    newY = self.p.y + (self.q.y - self.p.y) * t
                    return qd.Point(newX, newY)

        return None


if __name__ == '__main__':

    for i in range(0):
        p1 = qd.Point(rd.random(), rd.random())
        q1 = qd.Point(rd.random(), rd.random())
        seg1 = Segment(p1, q1, True)

        p2 = qd.Point(rd.random(), rd.random())
        q2 = qd.Point(rd.random(), rd.random())
        seg2 = Segment(p2, q2, True)

        intersection = seg1.intersection(seg2)

        if intersection:
            plt.plot(intersection.x, intersection.y, 'ro')

        drawLine(p1, q1)
        drawLine(p2, q2)

        plt.show()

    p1 = qd.Point(0, 2)
    q1 = qd.Point(2, 2)

    seg1 = Segment(p1, q1, True)

    p2 = qd.Point(1, 3)
    q2 = qd.Point(3, 3)
    seg2 = Segment(p2, q2, True)

    intersection = seg1.intersection(seg2)

    if intersection:
        plt.plot(intersection.x, intersection.y, 'ro')

    drawLine(p1, q1)
    drawLine(p2, q2)

    plt.show()

    p1 = qd.Point(0, 1)
    q1 = qd.Point(0, 2)

    seg1 = Segment(p1, q1, True)

    p2 = qd.Point(1, 1.5)
    q2 = qd.Point(1, 2.5)
    seg2 = Segment(p2, q2, True)

    intersection = seg1.intersection(seg2)

    if intersection:
        plt.plot(intersection.x, intersection.y, 'ro')

    drawLine(p1, q1)
    drawLine(p2, q2)

    plt.show()