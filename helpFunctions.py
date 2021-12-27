from matplotlib import pyplot as plt


def drawLine(p1, p2):
    x_values = [p1.x, p2.x]
    y_values = [p1.y, p2.y]
    plt.plot(x_values, y_values, 'black')
