import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import leastsq
import random
import math

__all__ = ['HoughLine', 'LeastLine', 'RansacLine']


class HoughLine:
    def __init__(self):
        pass

    def fitline_houghlines(self, edges):
        """

        :param edges:
        :return: lines: class 'numpy.ndarray': [[[rho, theta]] [[..]] .. [[..]]]
        """
        # 标准霍夫变换
        # 参数1：要检测的二值图（一般是阈值分割或边缘检测后的图）
        # 参数2：距离 ρ 的精度，值越大，考虑越多的线
        # 参数3：角度 θ 的精度，值越小，考虑越多的线
        # 参数4：累加数阈值，值越小，考虑越多的线 计算(r,θ) 累加数，累加数超过一定值后就认为在同一直线上（有一个阈值
        # 检测出来的是极坐标（rho, theta)
        lines = cv2.HoughLines(edges, 0.8, np.pi/180, 90)
        return lines

    def fitline_houghlinesP(self, edges):
        """

        :param edges:
        :return: lines: class 'numpy.ndarray': [[[x1 y1 x2 y2]] [[....]]..[[....]]]
        """
        # 概率霍夫直线变换
        # HoughLinesP直接给出了直线的断点，在画出线段的时候可以偷懒
        # minLineLength：最短长度阈值，比这个长度短的线会被排除
        # maxLineGap：同一直线两点之间的最大距离
        lines = cv2.HoughLinesP(edges, 0.8, np.pi/180, 150, minLineLength=20, maxLineGap=10)
        return lines

    def drawline_fitline_houghlines(self, lines, drawing):
        """

        :param lines: lines: class 'numpy.ndarray': [[[rho, theta]] [[..]] .. [[..]]]
        :param drawing:
        :return:
        """
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            drawing = cv2.line(drawing, (x1, y1), (x2, y2), (255, 0, 0), 1)
        return drawing

    def drawline_fitline_houghlinesP(self, lines, drawing):
        """

        :param lines: lines: class 'numpy.ndarray': [[[x1 y1 x2 y2]] [[....]]..[[....]]]
        :param drawing:
        :return:
        """
        for line in lines:
            line = line[0]
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            drawing = cv2.line(drawing, (x1, y1), (x2, y2), (255, 0, 0), 1)
        return drawing

# 目标函数y=sin(2πx) , 加上一个正太分布的噪音干扰，用多项式去拟合
def real_func(x):
    return np.sin(2 * np.pi * x)

# 多项式函数(拟合函数，也就是h(x))
# ps: numpy.poly1d([1,2,3])  生成  $1x^2+2x^1+3x^0$*
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差函数
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

# 添加正则项的残差函数
def residual_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))
    #print('ret', ret)
    return ret


class LeastLine:
    def __init__(self):
        pass

    def fitline_least_square(self, M):
        """
        使用最小二乘法拟合y=sin(2πx)
        :param M: 多项式的次数
        :return: list 拟合参数
        """
        # 随机初始化多项式参数
        # 生成p+1个随机数的列表，这样poly1d函数返回的多项式次数就是p
        p_init = np.random.rand(M + 1)
        #print('p_init:', p_init)
        # 最小二乘法
        # leastsq（）函数可以很快速地使用最小二乘法对数据进行拟合
        # 三个参数：误差函数、函数参数列表、数据点
        # p_lsp = leastsq(residuals_func, p_init, args=(x, y))
        p_lsp = leastsq(residual_func_regularization, p_init, args=(x, y))
        #print('Fitting Parameters:', p_lsp[0])

        # 可视化
        plt.plot(x_points, real_func(x_points), label='real')
        plt.plot(x_points, fit_func(p_lsp[0], x_points), label='fitted curve')
        plt.plot(x, y, 'bo', label='noise')
        plt.legend()
        plt.show()
        return p_lsp

class RansacLine:
    def __init__(self):
        pass

    def fitline_ransac(self, iters, sigma, SIZE, RANDOM_X, RANDOM_Y, P):
        # 最好模型的参数估计和内点数目
        best_a = 0
        best_b = 0
        pretotal = 0
        while iters > 0:
            # 随机在数据中红选出两个点去求解模型
            sample_index = random.sample(range(SIZE * 2), 2)
            x_1 = RANDOM_X[sample_index[0]]
            x_2 = RANDOM_X[sample_index[1]]
            y_1 = RANDOM_Y[sample_index[0]]
            y_2 = RANDOM_Y[sample_index[1]]

            # y = ax + b 求解出a，b
            a = (y_2 - y_1) / (x_2 - x_1)
            b = y_1 - a * x_1

            # 算出内点数目
            total_inlier = 0
            for index in range(SIZE * 2):
                y_estimate = a * RANDOM_X[index] + b
                if abs(y_estimate - RANDOM_Y[index]) < sigma:
                    total_inlier = total_inlier + 1

            # 判断当前的模型是否比之前估算的模型好
            if total_inlier > pretotal:
                iters = int(math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE * 2), 2)))
                #print(iters)
                pretotal = total_inlier
                best_a = a
                best_b = b

            # 判断是否当前模型已经符合超过一半的点
            #print("t:", total_inlier)
            if total_inlier > SIZE / 2:
                break
        return best_a, best_b


if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    # 已经进行初步的轮廓检测之后，才进行直线检测

    # Hough找直线
    drawing = np.zeros(img.shape[:], dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    hough = HoughLine()
    lines = hough.fitline_houghlinesP(edges)
    drawing = hough.drawline_fitline_houghlinesP(lines, drawing)


    plt.subplot(231), plt.imshow(img), plt.title('img')
    plt.subplot(232), plt.imshow(drawing), plt.title('drawing')
    plt.subplot(233), plt.imshow(gray), plt.title('gray')
    plt.subplot(234), plt.imshow(edges), plt.title('edges')
    plt.show()

    # 使用最小二乘法拟合y=sin(2πx)
    ll = LeastLine()
    # 加正则化项 正则化系数lambda不能过高，过高则导致欠拟合现象即此时的惩罚项权重较高
    regularization = 0.0001
    # 10个点 随机选取0-1之间的9个数作为x
    x = np.linspace(0, 1, 10)
    # 画图时需要的连续点
    x_points = np.linspace(0, 1, 1000)
    # 目标函数
    y_ = real_func(x)
    # 加上正态分布噪音的目标函数的值
    y = [np.random.normal(0, 0.1) + y1 for y1 in y_]
    p_lsp_0 = ll.fitline_least_square(M=9)

    # Random Sample consensus: 随机采样一致算法
    rc = RansacLine()
    # 数据量。
    SIZE = 50
    # 产生数据。np.linspace 返回一个一维数组，SIZE指定数组长度。
    # 数组最小值是0，最大值是10。所有元素间隔相等。
    X = np.linspace(0, 10, SIZE)
    #print(X)
    Y = 3 * X + 10

    fig = plt.figure()
    # 画图区域分成1行1列。选择第一块区域。
    ax1 = fig.add_subplot(1, 1, 1)
    # 标题
    ax1.set_title("RANSAC")

    # 让散点图的数据更加随机并且添加一些噪声。
    random_x = []
    random_y = []
    # 添加直线随机噪声
    for i in range(SIZE):
        random_x.append(X[i] + random.uniform(-0.5, 0.5))
        random_y.append(Y[i] + random.uniform(-0.5, 0.5))
    # 添加随机噪声
    for i in range(SIZE):
        random_x.append(random.uniform(0, 10))
        random_y.append(random.uniform(10, 40))
    RANDOM_X = np.array(random_x)  # 散点图的横轴。
    RANDOM_Y = np.array(random_y)  # 散点图的纵轴。

    # 画散点图。
    ax1.scatter(RANDOM_X, RANDOM_Y)
    # 横轴名称。
    ax1.set_xlabel("x")
    # 纵轴名称。
    ax1.set_ylabel("y")

    # 使用RANSAC算法估算模型
    # 迭代最大次数，每次得到更好的估计会优化iters的数值
    iters = 100000
    # 数据和模型之间可接受的差值
    sigma = 0.75
    # 希望的得到正确模型的概率
    P = 0.99
    best_a, best_b = rc.fitline_ransac(iters, sigma, SIZE, RANDOM_X, RANDOM_Y, P)
    Y = best_a * RANDOM_X + best_b
    # 直线图
    ax1.plot(RANDOM_X, Y)
    text = "best_a = " + str(best_a) + "\nbest_b = " + str(best_b)
    plt.text(5, 10, text,
             fontdict={'size': 8, 'color': 'r'})
    plt.show()