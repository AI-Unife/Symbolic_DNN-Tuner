import numpy as np
import matplotlib.pyplot as plt


def line_eq(x1, y1, x2, y2):
    m = float(y2 - y1) / (x2 - x1)
    q = y1 - (m * x1)

    def _line(x):
        return m * x + q

    return _line


def build_pendence(loss):
    x = []
    ip = []
    r = line_eq(0, loss[0], len(loss) - 1, loss[len(loss) - 1])

    for i in range(len(loss)):
        x.append(i)

    for j in x:
        ip.append(r(j))

    return ip


def integrals(loss):
    '''
    This function calculates the integral of the original loss function and his slope
    :param loss: original loss function
    :return: integral values of the original loss and slope respectively
    '''
    ip = build_pendence(loss)
    return np.trapz(loss), np.trapz(ip)


if __name__ == '__main__':
    loss = [1.5653950647735595, 1.2279206122207642, 1.1177522505569457, 1.0318743909835815, 0.9708976544189453,
            0.9248824000930786, 0.8893971458053589, 0.8470621564483642, 0.81564016248703, 0.7788446407699585,
            0.7601784302902221, 0.7296552336120605, 0.7171665460586548, 0.6943231027984619, 0.6729027723503113,
            0.6590187903404235, 0.6438528129959107, 0.6298836848831176, 0.6148716356277466, 0.6007654847335815,
            0.5859516922950745, 0.577476800956726, 0.5697370506381989, 0.5559622830772399, 0.5441177491569519,
            0.5350358511734009, 0.5287390618038178, 0.5224093583869934, 0.5105063557052613, 0.5012116564559936,
            0.5020502586460114, 0.4931805318260193, 0.483615428943634, 0.47568892528533935, 0.4717344103908539,
            0.4571272534942627, 0.45569468992233275, 0.45264851598739625, 0.4452364354324341, 0.4412414326572418,
            0.4325654611778259, 0.43957919874191287, 0.4319496483516693, 0.4272186267185211, 0.4147847042274475,
            0.4097470125198364, 0.4020730114746094, 0.40556993194580077, 0.3976480746650696, 0.39562178922653196,
            0.3913328070449829, 0.38608719902038574, 0.38687917662620547, 0.37460811292648316, 0.37353745193481447,
            0.3706480048179627, 0.3649155541324616, 0.3694863224220276]
    il, ip = integrals(loss)

    print(il)
    print(ip)