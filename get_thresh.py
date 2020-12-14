import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


if __name__ == "__main__":
    input_path = ''
    img = cv2.imread(input_path + 'ng_6_re.jpg')

    # 二值化处理
    # 返回 <class 'float'> 100.0 <class 'numpy.ndarray'> (398, 1190, 3)
    # 超过阈值为maxval，否则为0
    ret, thresh1 = cv2.threshold(img,
                                 thresh=100,
                                 maxval=80,
                                 type=cv2.THRESH_BINARY)
    # 超过阈值为0，否则为maxval（相当于上个参数取反）
    ret, thresh2 = cv2.threshold(img,
                                 thresh=100,
                                 maxval=80,
                                 type=cv2.THRESH_BINARY_INV)
    # 超过阈值为thresh，低于阈值灰度值不变
    ret, thresh3 = cv2.threshold(img,
                                 thresh=200,
                                 maxval=255,
                                 type=cv2.THRESH_TRUNC)
    # 超过阈值灰度值不变，否则为0
    ret, thresh4 = cv2.threshold(img,
                                 thresh=110,
                                 maxval=255,
                                 type=cv2.THRESH_TOZERO)
    # 超过阈值为0，低于阈值灰度值不变
    ret, thresh5 = cv2.threshold(img,
                                 thresh=110,
                                 maxval=255,
                                 type=cv2.THRESH_TOZERO_INV)

    # 自动二值化处理
    img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu阈值
    ret, thresh6 = cv2.threshold(img_0,
                                 thresh=0,
                                 maxval=255,
                                 type=cv2.THRESH_OTSU)
    # Otsu阈值
    ret, thresh7 = cv2.threshold(img_0,
                                 thresh=0,
                                 maxval=255,
                                 type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    # Triangle
    ret, thresh8 = cv2.threshold(img_0,
                                 thresh=0,
                                 maxval=255,
                                 type=cv2.THRESH_TRIANGLE)

    # 自适应二值化
    # maxval为像素值的最大值默认设为255
    # blockSize为计算像素阈值的像素邻域的大小：3、5、7。。。。、99
    # C为从平均值或加权平均值中减去常数，一般设置为0-50
    # 高斯BINARY
    thresh9 = cv2.adaptiveThreshold(img_0,
                                    maxValue=255,
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType=cv2.THRESH_BINARY,
                                    blockSize=25,
                                    C=5)
    # 高斯BINARY_INV
    thresh10 = cv2.adaptiveThreshold(img_0,
                                    maxValue=255,
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType=cv2.THRESH_BINARY_INV,
                                    blockSize=3,
                                    C=5)
    # 平均BINARY
    thresh11 = cv2.adaptiveThreshold(img_0,
                                    maxValue=255,
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                    thresholdType=cv2.THRESH_BINARY,
                                    blockSize=3,
                                    C=5)
    # 平均BINARY_INV
    thresh12 = cv2.adaptiveThreshold(img_0,
                                    maxValue=255,
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                    thresholdType=cv2.THRESH_BINARY_INV,
                                    blockSize=3,
                                    C=5)
    plt.subplot(331), plt.imshow(img), plt.title('img')
    plt.subplot(332), plt.imshow(thresh1), plt.title('thresh1')
    plt.subplot(333), plt.imshow(thresh2), plt.title('thresh2')
    plt.subplot(334), plt.imshow(thresh3), plt.title('thresh3')
    plt.subplot(335), plt.imshow(thresh4), plt.title('thresh4')
    plt.subplot(336), plt.imshow(thresh5), plt.title('thresh5')
    plt.subplot(337), plt.imshow(thresh6), plt.title('thresh6')
    plt.subplot(338), plt.imshow(thresh7), plt.title('thresh7')
    plt.subplot(339), plt.imshow(thresh8), plt.title('thresh8')
    plt.show()
    plt.subplot(221), plt.imshow(thresh9), plt.title('thresh9')
    plt.subplot(222), plt.imshow(thresh10), plt.title('thresh10')
    plt.subplot(223), plt.imshow(thresh11), plt.title('thresh11')
    plt.subplot(224), plt.imshow(thresh12), plt.title('thresh12')
    plt.show()