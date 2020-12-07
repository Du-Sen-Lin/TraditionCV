import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time

__all__ = ['ColorReplace']
input_path = ''
out_path = ''


class ColorReplace():
    def __init__(self, input_path=None):
        """
        :param input_path:images path
        """
        self.path = input_path

    def replace(self, img, dst_clr):
        """
        replace image's pixel in specific area
        :param img: input image
        :param dst_clr: pixel to replace
        :return: img: images : np.array
        """
        for i in range(80, 340): #x1 x2
            for j in range(500, 800): #y1 y2
                img[j][i] = dst_clr
        return img

    def replace_fast(self, img, dst_clr):
        """
        fast replace image's pixel in specific area
        :param img:input image
        :param dst_clr:pixel to replace
        :return: img: images : np.array
        """
        img[535:750, :290, :] = dst_clr #h(y) w(x) c
        img[575:705, 900:, :] = dst_clr
        return img

    def read_write_img(self):
        """
        write images after replace ..
        :return:
        """
        for file in os.listdir(self.path):
            filelist = input_path + file
            img = cv2.imread(filelist)
            dst_img = self.replace_fast(img, (0, 0, 0))
            # cv2.imwrite(out_path + file[:-4] + '.jpg', re_img)
            plt.subplot(121), plt.imshow(img), plt.title('initial')
            plt.subplot(122), plt.imshow(dst_img), plt.title('result')
            plt.show()


if __name__ == "__main__":
    cr = ColorReplace(input_path=input_path)
    cr.read_write_img()