import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input, Add
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os

dirpath1 = "D:\\ML\\images\\test\\Littleface\\f\\"
dirpath2 = "D:\\ML\\images\\test\\Littleface\\t\\"

names1 = os.listdir(dirpath1)
names2 = os.listdir(dirpath2)

def getNames(names1, names2):
    l = len(names1)
    if l != len(names2):
        return

    for i in range(l):
        yield names1[i], names2[i]

# myInput = Input(shape=(55, 47, 3))
# x = Conv2D(20, (4, 4), name='Conv1', activation='relu', input_shape=(55, 47, 3))(myInput)
# x = MaxPooling2D(pool_size=2, strides=2, name='Pool1')(x)
# x = Dropout(rate=0.1, name='D1')(x)
#
# x = Conv2D(40, (3, 3), name='Conv2', activation='relu')(x)
# x = MaxPooling2D(pool_size=2, strides=2, name='Pool2')(x)
# x = Dropout(rate=0.1, name='D2')(x)
#
# x = Conv2D(60, (3, 3), name='Conv3', activation='relu')(x)
# x = MaxPooling2D(pool_size=2, strides=2, name='Pool3')(x)
# x = Dropout(rate=0.1, name='D3')(x)
#
# x1 = Flatten()(x)
# fc11 = Dense(160, name='fc11')(x1)
#
# x2 = Conv2D(80, (2, 2), name='Conv4', activation='relu')(x)
# x2 = Flatten()(x2)
# fc12 = Dense(160, name='fc12')(x2)
#
# y = Add()([fc11, fc12])
# y = Activation('relu', name='deepid')(y)
#
# model = Model(inputs=[myInput], outputs=y)
# model.summary()
#
# model.load_weights(r"D:\ML\model\FaceVerification\deepid_keras_weights.h5")

# model = keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# model = keras.Model(model.input, model.get_layer('avg_pool').output)
# model = keras.applications.vgg16.VGG16()
# model = keras.Model(model.input, model.get_layer('fc2').output)
# model = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
# model = keras.Model(model.input, model.get_layer('reshape_2').output)
model = keras.models.load_model(r"D:\ML\model\FaceVerification\mobilev2_faceV_02")

# ==========================================================
# IMAGE_SIZE = (55, 47)
IMAGE_SIZE = (224, 224)
img1_path = r"D:\ML\images\Face\vggface_60\n000213\0001_01.jpg"
img2_path = r"D:\ML\images\Face\vggface_60\n000213\0002_01.jpg"

from FaceVerification.getFace import getFace
GF = getFace()


def CompareFace(img1_path, img2_path):
    img1 = GF.load_faces(img1_path, IMAGE_SIZE)
    img2 = GF.load_faces(img2_path, IMAGE_SIZE)

    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()

    img1_representation = model.predict(tf.expand_dims(img1, 0))[0,:]
    img2_representation = model.predict(tf.expand_dims(img2, 0))[0,:]

    def findCosineDistance(source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    cosine_distance = findCosineDistance(img1_representation, img2_representation)
    # euclidean_distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    # euclidean_l2_distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
    #                                                   dst.l2_normalize(img2_representation))

    print(cosine_distance)
    if cosine_distance>0.03:
        print("Both face image not one people")
    else:
        print("Yes, Both face is one people")

# while 1:
#     img1_path = input('img1')
#     img2_path = input('img2')

# for (i, j) in getNames(names1, names2):
#     print(i, j)
#     CompareFace(dirpath1 + i, dirpath2+j)

CompareFace(img1_path, img2_path)