import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

__all__ = ['FindContours']


class FindContours:
    def __init__(self, input_path):
        """
        :param input_path: images path
        """
        self.path = input_path

    def find_contours(self, img):
        """
        find all contours in image
        :param img: images :np.array
        :return: contours : all contours :List
        """
        blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = cv2.Sobel(blur, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)
        Scale_absX = cv2.convertScaleAbs(x)
        Scale_absY = cv2.convertScaleAbs(y)
        sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        ret, im_fixed = cv2.threshold(sobel, 20, 255, cv2.THRESH_BINARY)

        # 第一个参数是寻找轮廓的图像；
        # 第二个参数表示轮廓的检索模式，有四种
        #     cv2.RETR_EXTERNAL表示只检测外轮廓
        #     cv2.RETR_LIST检测的轮廓不建立等级关系
        #     cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        #     cv2.RETR_TREE建立一个等级树结构的轮廓。
        # 第三个参数method为轮廓的近似办法
        #     cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        #     cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        #     cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
        # cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。
        contours, hierarchy = cv2.findContours(im_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def find_max_contours(self, img):
        """
        find the max contour in all contours
        :param img:images : np.array
        :return: contours[num]: the max contours : list
        """
        contours = self.find_contours(img)
        max = 0
        num = -1
        for i in range(0, len(contours)):
            if len(contours[i]) > max:
                max = len(contours[i])
                num = i
        return contours[num]

    def draw_contours(self, img, contour, a=-1, rgb=(255, 0, 255), thickness=3):
        """
        draw contours
        :param img: images : np.array
        :param contour: contours : list :if np.array,add [] to list
        :param a: contourIdx
        :param rgb: color
        :param thickness: thickness
        :return: img: np.ndarray
        """
        # 第一个参数是指明在哪幅图像上绘制轮廓；image为三通道才能显示轮廓
        # 第二个参数是轮廓本身，在Python中是一个list;
        # 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓
        # 第四个参数： 轮廓颜色
        # 第五个参数： thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。
        # ...
        img = cv2.drawContours(img, [contour], a, rgb, thickness)
        return img

    def get_moments(self, contour):
        """
        get centroid of object(对象质心)
        :param contour: np.ndarray
        :return: cx,cy : centroid of object(对象质心) by images moments infomation :int
        """
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy

    def get_contourArea(self, contour):
        """
        contour area.
        :param contour: np.ndarray
        :return: area : float
        """
        area = cv2.contourArea(contour)
        return area

    def get_contourLength(self, contour):
        perimeter = cv2.arcLength(contour, True)
        return perimeter

    def draw_rectangle(self, img, contour):
        """
        draw rectangle according to contour
        :param img: np.ndarray
        :param contour: np.ndarray
        :return: np.ndarray
        """
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
        return img

    def draw_min_rectangle(self, img, contour):
        """
        draw min rectangle
        :param img:
        :param contour:
        :return:
        """
        rect = cv2.minAreaRect(contour)
        rect = np.int0(cv2.boxPoints(rect))  # 返回四个点的值 np.int0(x) 取整
        img = cv2.drawContours(img, [rect], 0, (255, 0, 0), 3)
        return img


    def draw_min_circle(self, img, contour):
        """

        :param img:
        :param contour:
        :return:
        """
        (x, y), radius = cv2.minEnclosingCircle(contour)
        (x, y, radius) = np.int0((x, y, radius))
        img = cv2.circle(img, (x, y), radius, (0, 255, 0), 3)
        return img

    def draw_ellipse(self, img, contour):
        """

        :param img:
        :param contour:
        :return:
        """
        ellipse = cv2.fitEllipse(contour)
        img = cv2.ellipse(img, ellipse, (255, 255, 0), 3)
        return img

    def draw_approximation_polygon(self, img, contour, fine=3, flg=True):
        """

        :param img:
        :param contour:
        :param fine:
        :param flg:
        :return:
        """
        # 多边形逼近(参数2表示距离值，表示多边形的轮廓接近实际轮廓的程度，值越小，越精确，参数3表示是否闭合）
        approx = cv2.approxPolyDP(contour, 3, True)
        img = cv2.polylines(img, [approx], True, (255, 255, 255), 3)
        return img

    def draw_convex_hull(self, img, contour):
        """

        :param img:
        :param contour:
        :return:
        """
        # 凸包(函数cv2.convexHull()有个可选参数returnPoints，默认是True，代表返回角点的x/y坐标；如果为False的话，表示返回轮廓中是凸包角点的索引)
        hull = cv2.convexHull(contour)
        img = cv2.polylines(img, [hull], True, (0, 0, 0), 3)
        flg = cv2.isContourConvex(hull) #返回轮廓是否为凸性的，是为True，否为False
        return img, flg

    def read_write_img(self):
        """
        test
        :return:
        """
        img = cv2.imread(self.path + '0012.jpg')
        img_0 = img.copy()
        img_1 = img.copy()
        img_2 = img.copy()

        # test start
        # 最大轮廓
        max_contours = self.find_max_contours(img)
        # 画轮廓
        img_0_drawcontours = self.draw_contours(img_0, contour=max_contours)
        # 得到轮廓质心（重心）
        cx, cy = self.get_moments(max_contours)
        print('centroid of object:',   cx, cy)
        # 得到轮廓面积
        area = self.get_contourArea(max_contours)
        print('area:', area)
        # 得到轮廓长度
        perimeter = self.get_contourLength(max_contours)
        print('length:', perimeter)
        # 画外接矩形
        img_0_drawrectangle = self.draw_rectangle(img_0, contour=max_contours)
        # 画最小外接矩形
        img_0_drawminrectangle = self.draw_min_rectangle(img_0, contour=max_contours)
        # 画最小外接圆
        img_0_drawmincircle = self.draw_min_circle(img_0, contour=max_contours)
        # 画拟合椭圆
        img_0_drawellipse = self.draw_ellipse(img_0, contour=max_contours)
        # 多边形逼近
        img_0_draw_approximation_polygon = self.draw_approximation_polygon(img_0, contour=max_contours, fine=3, flg=True)
        # 凸包 是否凸性
        img_1_drawconvexhull, flg = self.draw_convex_hull(img_1, contour=max_contours)
        print('convex_hull: ', flg)
        # test end

        plt.subplot(331), plt.imshow(img), plt.title('images')
        plt.subplot(332), plt.imshow(img_0_drawcontours), plt.title('img_0_drawcontours')
        plt.subplot(333), plt.imshow(img_0_drawrectangle), plt.title('img_0_drawrectangle')
        plt.subplot(334), plt.imshow(img_0_drawminrectangle), plt.title('img_0_drawminrectangle')
        plt.subplot(335), plt.imshow(img_0_drawmincircle), plt.title('img_0_drawmincircle')
        plt.subplot(336), plt.imshow(img_0_drawellipse), plt.title('img_0_drawellipse')
        plt.subplot(337), plt.imshow(img_0_draw_approximation_polygon), plt.title('img_0_draw_approximation_polygon')
        plt.subplot(338), plt.imshow(img_1_drawconvexhull), plt.title('img_1_drawconvexhull')
        plt.show()

        return max_contours


if __name__ == "__main__":
    input_path = ''
    fc = FindContours(input_path=input_path)
    re = fc.read_write_img()