import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time


savepath1 = "E:/Project/bp/defect1/"


def postprocess(img, filename, area_threshold=80, denoising_params=(10, 15, 7, 21), canny_params=(110, 190),
                threshold_params=(235, 255), morph_kernel_size=(3, 3), dilate_iterations=3, erode_iterations=2):
    """
    目标检测结果后处理
    :param img: 目标检测缺陷区域
    :param area_threshold: 单个轮廓过滤阈值
    :param denoising_params: 去噪参数 (h, hColor, templateWindowSize, searchWindowSize)
    :param canny_params: Canny边缘检测参数 (lower_threshold, upper_threshold)
    :param threshold_params: 二值化阈值参数 (threshold, max_value)
    :param morph_kernel_size: 形态学操作核的大小 (rows, cols)
    :param dilate_iterations: 膨胀操作迭代次数
    :param erode_iterations: 腐蚀操作迭代次数
    :return: area_total 超过设定阈值的轮廓面积和，用于阈值开发接口设定值过滤
    """
    s = time.time()
    area_total = 0
    # blur = cv2.fastNlMeansDenoisingColored(img, None, *denoising_params)
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    # canny = cv2.Canny(blur, *canny_params)
    canny = cv2.Canny(blur, 120, 200)
    ret, im_fixed = cv2.threshold(canny, *threshold_params, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    dilated = cv2.dilate(im_fixed, kernel, iterations=dilate_iterations)
    erode = cv2.erode(dilated, kernel, iterations=erode_iterations)
    contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            area_total += area

            cv2.drawContours(
                contour_image, [contour], -1, 255, thickness=3)
            x, y, w, h = cv2.boundingRect(contour)
            contour_image = cv2.rectangle(
                contour_image, (x, y), (x+w, y+h), (0, 255, 255), 3)
    # cv2.imwrite(savepath1 + filename, contour_image)

    plt.subplot(331), plt.imshow(img), plt.title('img')
    plt.subplot(332), plt.imshow(blur), plt.title('blur')
    plt.subplot(333), plt.imshow(canny), plt.title('canny')
    plt.subplot(334), plt.imshow(im_fixed), plt.title('im_fixed')
    plt.subplot(335), plt.imshow(dilated), plt.title('dilated')
    plt.subplot(336), plt.imshow(erode), plt.title('erode')
    plt.subplot(337), plt.imshow(contour_image), plt.title('contour_image')
    plt.show()
    print(f"time: {time.time() - s}.")
    return area_total


if __name__ == "__main__":
    # filename2 = 'E:/Project/bp/crop/22190081-1670223011367_s1_HS0.jpg'
    for filename in os.listdir((r"E:/Project/bp/crop")):
        filename2 = 'E:/Project/bp/crop/' + filename
        img = cv2.imread(filename2, 1)
        area_total = postprocess(img, filename)
        print(f"area_total: {area_total}")