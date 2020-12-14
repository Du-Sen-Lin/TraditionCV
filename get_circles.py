import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from get_color_replace import ColorReplace

__all__ = ['FindCircles']
input_path = ''
out_path = ''


class FindCircles:
    def __init__(self, input_path=None):
        """
        :param input_path: images path
        """
        self.path = input_path

    def defect_circle(self, img):
        """
        defect circles in images by HoughCircles
        :param img: input image
        :return: circles: [[[center_coor_x, center_coor_y, radius] [...]...]] :list
        """
        # image:输入图像 (灰度图)
        # method:指定检测方法. 现在OpenCV中只有霍夫梯度法
        # dp:累加器图像的反比分辨=1即可默认  2特别强
        # 检测内侧圆心的累加器图像的分辨率于输入图像之比的倒数，如dp=1，
        # 累加器和输入图像具有相同的分辨率，如果dp=2，累计器便有输入图像一半那么大的宽度和高度
        # minDist = src_gray.rows/8: 检测到圆心之间的最小距离，这是一个经验值。这个大了，那么多个圆就是被认为一个圆。
        # param_1 = 200: Canny边缘函数的高阈值 而低阈值为高阈值的一半
        # param_2 = 100: 圆心检测阈值.根据你的图像中的圆大小设置，当这张图片中的圆越小，那么此值就设置应该被设置越小。
        # 当设置的越小，那么检测出的圆越多，在检测较大的圆时则会产生很多噪声。所以要根据检测圆的大小变化
        # min_radius = 0: 能检测到的最小圆半径, 默认为0.
        # max_radius = 0: 能检测到的最大圆半径, 默认为0
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 600, param1=150,
                                   param2=20, minRadius=10, maxRadius=150)
        return circles

    def read_img(self):
        """
        read images in file path
        :return: img_tmp: draw circles image : np.array
        """
        img = cv2.imread(self.path + '10000.jpg')
        img_tmp = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = self.defect_circle(gray)
        print('circles nums:', len(circles[0]))
        circles = circles[0, :, :]
        circles = np.uint16(np.around(circles))
        print('circles:', circles)
        for i in circles[:]:
            cv2.circle(img_tmp, (i[0], i[1]), i[2], (255, 0, 0), 5)
            cv2.circle(img_tmp, (i[0], i[1]), 2, (255, 0, 255), 10)
            cv2.rectangle(img_tmp, (i[0] - i[2], i[1] + i[2]),
                          (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)
            print("center_coor and r:", i[0], i[1], i[2], end=' ')
        print('\n')
        return img_tmp


if __name__ == "__main__":
    fc = FindCircles(input_path=input_path)
    re = fc.read_img()
    plt.subplot(121),plt.imshow(re),plt.title('circles')
    plt.show()
    cr = ColorReplace('')
    cr.read_write_img()