import numpy as np
import cv2
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 调整最大值
MAX_VALUE = 100
dataset_dir = r"D:\ML\images\Face\Celeb_a\Humans"
output_dir = r"D:\ML\images\test\Littleface\cccc"

# lightness =  -0.6  #int(input("lightness(亮度-100~+100):"))  # 亮度
# saturation = 1    #int(input("saturation(饱和度-100~+100):"))  # 饱和度

def update(input_img_path, lightness, saturation, output_img_path = None):
    """
    用于修改图片的亮度和饱和度
    :param input_img_path: 图片路径
    :param output_img_path: 输出图片路径
    :param lightness: 亮度
    :param saturation: 饱和度
    """
    # 加载图片 读取彩色图像归一化且转换为浮点型
    img = cv2.imdecode(np.fromfile(input_img_path, dtype=np.uint8), 1)/255.0
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, saturation, blank, 1 - saturation, lightness)

    return dst

def dir2dir(srcDir, dstDir, lightness=-0.6, saturation=1):
    num = 0
    for img_name in os.listdir(dataset_dir)[:625]:
        srcPath = srcDir + '/' + img_name
        dstPath = dstDir + '/' + img_name
        shutil.copy(srcPath, dstPath)
        # result = update(srcPath, lightness, saturation)
        # cv2.imwrite(dstPath, result*255.0)
        print(num, ':', srcPath, '-->', dstPath)

# dir2dir(dataset_dir, output_dir)

srcPath = r"D:\ML\images\test\Littleface\enhence\high\1 (100).png"
result = update(srcPath, -0.6, 1)*255.0
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

img = plt.imread(srcPath)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(result)
plt.show()
