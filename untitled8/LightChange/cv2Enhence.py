import numpy as np # linear algebra
import os
import cv2
import matplotlib.pyplot as plt

img_path = r"D:\ML\images\test\Littleface\enhence\low\1 (1011).jpg"
IMAGE_SHAPE = (256, 256)


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def ProcessImage(ImagePath):
    size = 3
    img = cv2.imread(ImagePath)

    b_gray, g_gray, r_gray = cv2.split(img)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(result)

    plt.show()

ProcessImage(img_path)