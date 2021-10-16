import numpy as np # linear algebra
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape

InputPath= "D:/ML/images/BrighteningTrain/"
IMAGE_SHAPE = (256, 256)
EPOCHES = 10
IMAGE_NUMBER = 501

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

def PreProcessData(ImagePath):
    X_ = []
    y_ = []
    count = 0
    size = 3

    for imageDir in os.listdir(ImagePath):
        print(imageDir)
        if count < IMAGE_NUMBER:
            try:
                count = count + 1
                img = cv2.imread(ImagePath + imageDir)

                b_gray, g_gray, r_gray = cv2.split(img)
                b_gray = SSR(b_gray, size)
                g_gray = SSR(g_gray, size)
                r_gray = SSR(r_gray, size)
                result = cv2.merge([b_gray, g_gray, r_gray])

                X_.append(img)
                y_.append(result)

                if count % 10 == 0:
                    print(count)

            except:
                pass
    X_ = np.array(X_)
    y_ = np.array(y_)

    return X_, y_

def PreProcessData2(ImagePath):
    high_path = ImagePath + 'high' + '/'
    low_path = ImagePath + 'low' + '/'
    highNames = sorted(os.listdir(low_path))
    lowNames = sorted(os.listdir(high_path))
    X_ = []
    y_ = []
    count = 0

    for img_name in lowNames:
        img = cv2.imread(low_path + img_name)
        X_.append(img)

        count += 1
        if count%40==0:
            print("count:", count)
    for img_name in highNames:
        img = cv2.imread(high_path + img_name)
        y_.append(img)

        count += 1
        if count%40==0:
            print("count:", count)

    return X_, y_

def InstantiateModel(in_):
    model_1 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=1)(in_)
    model_1 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=1)(model_1)
    model_1 = Conv2D(64, (2, 2), activation='relu', padding='same', strides=1)(model_1)

    model_2 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=1)(in_)
    model_2 = Conv2D(64, (2, 2), activation='relu', padding='same', strides=1)(model_2)

    model_2_0 = Conv2D(64, (2, 2), activation='relu', padding='same', strides=1)(model_2)

    model_add = add([model_1, model_2, model_2_0])

    model_3 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1)(model_add)
    model_3 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=1)(model_3)
    model_3 = Conv2D(16, (2, 2), activation='relu', padding='same', strides=1)(model_3)

    model_3_1 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=1)(model_add)
    model_3_1 = Conv2D(16, (2, 2), activation='relu', padding='same', strides=1)(model_3_1)

    model_3_2 = Conv2D(16, (2, 2), activation='relu', padding='same', strides=1)(model_add)

    model_add_2 = add([model_3_1, model_3_2, model_3])

    model_4 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=1)(model_add_2)
    model_4_1 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=1)(model_add)
    # Extension
    model_add_3 = add([model_4_1, model_add_2, model_4])

    model_5 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=1)(model_add_3)
    model_5 = Conv2D(16, (2, 2), activation='relu', padding='same', strides=1)(model_add_3)

    model_5 = Conv2D(3, (3, 3), activation='relu', padding='same', strides=1)(model_5)

    return model_5

# 搭建生成器
def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = tf.image.resize(X[i]/255.0, IMAGE_SHAPE)
        y_input = tf.image.resize(y[i]/255.0, IMAGE_SHAPE)
        yield (tf.expand_dims(X_input, 0), tf.expand_dims(y_input, 0))

def preprocess(x, y):  # 自定义的预处理函数
    # 调用此函数时会自动传入 x,y 对象，shape 为[b, 28, 28], [b]
    # 标准化到 0~1

    x = tf.image.resize(x, IMAGE_SHAPE)/255.0
    y = tf.image.resize(y, IMAGE_SHAPE)/255.0
    # 返回的 x,y 将替换传入的 x,y 参数，从而实现数据的预处理功能
    return x, y

X_,y_ = PreProcessData2(InputPath)

# ===============================================================
# print(X_.shape, y_.shape)).batch(32).repeat()

#
# train_db = tf.data.Dataset.from_tensor_slices((X_,y_))#构建Dataset对象
# train_db = train_db.map(preprocess)
#
# train_data = train_db.shuffle(1000
# for a, b in train_data:
#     print(a.shape)
#     print(b.shape)
#     break


# print(X_)
# print(y_.shape)

# for x, y in GenerateInputs(X_,y_):
#     print(x.shape)
#     plt.imshow(x)
#     plt.imshow(y)
#     plt.show()
#     break



# plt.imshow(y)
# plt.show()
# ================================================


# Input_Sample = Input(shape= IMAGE_SHAPE +(3,))
# Output_ = InstantiateModel(Input_Sample)
# Model_Enhancer = keras.Model(inputs=Input_Sample, outputs=Output_)
# Model_Enhancer = keras.models.load_model(r"D:\ML\model\My_test\lowLight_sceneryEnhancer_05")
# Model_Enhancer.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss='mean_squared_error')
# Model_Enhancer.summary()


# 训练增强器
# Model_Enhancer.fit_generator(GenerateInputs(X_, y_), epochs=EPOCHES,
#                              steps_per_epoch= 300, shuffle=True)
# Model_Enhancer.save(r"D:\ML\model\My_test\lowLight_sceneryEnhancer_05")

# 训练减弱器
# Model_Enhancer.fit_generator(GenerateInputs(y_, X_), epochs=EPOCHES,
#                              steps_per_epoch= 10, shuffle=True)
# Model_Enhancer.save(r"D:\ML\model\My_test\lowLight_wakener_01")
