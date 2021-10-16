import numpy as np # linear algebra
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf

IMAGE_SHAPE = (256, 256)
image_path = r"D:\ML\images\test\Littleface\enhence\high\1 (100).png"
enchancer_model = keras.models.load_model(r"D:\ML\model\My_test\lowLight_sceneryWakener_02")
enchancer_model.summary()

# keras.utils.plot_model(enchancer_model, to_file='model1.png')
# weakener_model = keras.models.Sequential()
weakener_model = keras.models.load_model(r"D:\ML\model\My_test\lowLight_sceneryWakener_02")


def getImg_1per255(path):  
    image = cv2.imread(path)
    image = tf.image.resize(image/255.0, IMAGE_SHAPE)
    image = tf.expand_dims(image, 0)
    return image

def enchancer(img):
    # model = keras.models.load_model(r"D:\ML\model\My_test\lowLight_enhancer_02")
    result = enchancer_model.predict(img)
    print(result.shape)

    return (img, result)

def weakener(img):
    # model = keras.models.load_model(r"D:\ML\model\My_test\lowLight_sceneryWakener_01")
    result = weakener_model.predict(img)
    print(result.shape)

    return (img, result)


def showImg(image_path):
    image = getImg_1per255(image_path)

    # ===========在这里切换增强器和减弱器
    img, result = weakener(image)
    # cv2读取方式为BGR， plt显示方式为RGB，所以进行如下修改
    img = np.array(img)
    b, g, r = cv2.split(img[0])
    img[0] = cv2.merge([r, g, b])

    result = np.array(result)
    b, g, r = cv2.split(result[0])
    result[0] = cv2.merge([r, g, b])
    # ========================

    plt.figure()
    plt.subplot(121)
    plt.imshow(img[0])

    plt.subplot(122)
    plt.imshow(result[0])

    plt.show()


# img, result = weakener(image)

showImg(image_path)