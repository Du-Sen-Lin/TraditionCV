import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from get_color_replace import ColorReplace

__all__ = ['FindCircles']
input_path = 'C:/wood/work/SEMW/SEMW_switch/dataset/dataset/data_test/module_101/side_crop/'
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
    cr = ColorReplace('C:/wood/work/SEMW/SEMW_switch/dataset/dataset/data_test/module_101/front_ok_test/')
    cr.read_write_img()