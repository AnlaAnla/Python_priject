import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense


IMAGE_SIZE = (150, 150)
data_dir = r"D:\ML\images\newLeaf"
test_dir = r"D:\ML\images\newLeaf"

epochs = 100
freq = 10

class_names = os.listdir(data_dir)


def softmax_T(x, axis=1, T=1.5):
    # T 为温度系数
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x/T)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def AlexNet(path = r"D:\ML\model\ALL_leaf\LeafAlex_1"):
    model_alex=Sequential(name='AlexNet')
    model_alex.add(keras.layers.InputLayer(input_shape=(150, 150)+(3,)))
    model_alex.add(Conv2D(96,(11,11),strides=(4,4),padding='valid',activation='relu',kernel_initializer='uniform'))
    model_alex.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model_alex.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model_alex.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model_alex.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model_alex.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model_alex.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model_alex.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model_alex.add(Flatten())
    model_alex.add(Dense(1024,activation='relu'))
    model_alex.add(Dense(1024,activation='relu'))
    model_alex.add(Dense(16,activation='softmax'))
    model_alex.load_weights(path)
    return model_alex
def Vgg16(path = r"D:\ML\model\ALL_leaf\LeafVgg16_1"):
    img_input = layers.Input(shape=(150, 150) + (3,))
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(1024, activation='relu', name='fc1')(x)
    x = layers.Dense(1024, activation='relu', name='fc2')(x)
    x = layers.Dense(16, activation='softmax',
                     name='predictions')(x)

    inputs = img_input
    # Create model.
    model_vgg = keras.Model(inputs, x, name='vgg16')
    model_vgg.load_weights(path)
    return model_vgg

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    # subset='training'
    class_mode='categorical'
)
print(len(train_generator))
test_generator = test_datagen.flow_from_directory(
    test_dir, # same directory as training data
    target_size=(150, 150),
    batch_size=32,
    # subset='validation'，
    class_mode='categorical'
) # set as validation data
print(len(test_generator))


# for image_batch, label_batch in train_generator:
#     print(image_batch.shape, label_batch.shape)
#     break
def DistillVgg16():
    graphSava_path = r"D:\ML\model\ALL_leaf\redistill_LeafVgg16_2.txt"
    modelSave_path = r"D:\ML\model\ALL_leaf\redistill_leafVgg16_3"
    model_vgg = Vgg16()
    class MyTrainData:
        def __iter__(self):
            self.a = train_generator
            return self

        def __next__(self):
            for image_batch, label_batch in train_generator:
                break
            # self.a = next(train_generator)
            # x = self.a

            result = model_vgg.predict(image_batch)
            return (image_batch, result)
            # return (image_batch, label_batch)

    class MyTestData:
        def __iter__(self):
            self.a = train_generator
            return self

        def __next__(self):
            for image_batch, label_batch in train_generator:
                break
            # self.a = next(train_generator)
            # x = self.a
            return (image_batch, label_batch)

    model = keras.Sequential()
    model.add(layers.Input(shape=(150, 150) + (3,)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(layers.MaxPooling2D((2, 2), name='block1_pool'))
    # Block 2
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.MaxPooling2D((2, 2), name='block2_pool'))
    # Block 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(64, (2, 2), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.MaxPooling2D((2, 2), name='block3_pool'))

    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(128, activation='relu', name='fc1'))
    model.add(layers.Dense(128, activation='relu', name='fc2'))
    model.add(layers.Dense(16, activation='softmax', name='predictions'))
    # model = keras.models.load_model("D:\ML\model\ALL_leaf\distill_leafVgg16_1.h5")


    print(model.summary())



    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    history = model.fit_generator(MyTrainData(),
                                  steps_per_epoch=len(train_generator),
                                  epochs=epochs,
                                  validation_data=MyTestData(),
                                  validation_steps=len(test_generator),
                                  validation_freq=freq)
    model.save(filepath=modelSave_path)
    # print(history)

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs_range = range(epochs)

    # leaf_data1 = {"epochs_range": epochs,
    #               "acc": acc,
    #               "val_acc": val_acc,
    #               "loss": loss,
    #               "val_loss": val_loss}
    #
    # # 数据线的保存和读取
    # # Save
    # f = open(graphSava_path, 'w')
    # f.write(str(leaf_data1))
    # f.close()
    #
    # # Load
    # # 读取
    # f = open(graphSava_path, 'r')
    # a = f.read()
    # dict_name = eval(a)
    # f.close()
    # print(dict_name)

def DistillAlexNet():
    graphSava_path = r"D:\ML\model\ALL_leaf\distill_LeafAlex_2"
    modelSave_path = r"D:\ML\model\ALL_leaf\distill_leafAlex_2.h5"
    model_alex = AlexNet()
    class MyTrainData:
        def __iter__(self):
            self.a = train_generator
            return self
            #         .predict(image_batch)
            #         return (image_batch, result)
            #         # return (image_batch, label_batch)
            #
            #     class MyTestData:
            #         def __iter__(self):
            #             self.a = train_generator
            #             return self
            #
            #         def __next__(self):

        def __next__(self):
            for image_batch, label_batch in train_generator:
                break
            # self.a = next(train_generator)
            # x = self.a

            result = model_alex
            for image_batch, label_batch in train_generator:
                break
            # self.a = next(train_generator)
            # x = self.a
            return (image_batch, label_batch)
    class MyTestData:
        def __iter__(self):
            self.a = train_generator
            return self

        def __next__(self):
            for image_batch, label_batch in train_generator:
                break
            # self.a = next(train_generator)
            # x = self.a
            return (image_batch, label_batch)


    # model = keras.Sequential(name='leaf_smallModel')
    # model.add(keras.layers.InputLayer(input_shape=IMAGE_SIZE+(3,)))
    # model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    #
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(16 , name='logits'))
    # model.add(keras.layers.Softmax())
    model = keras.models.load_model("D:\ML\model\ALL_leaf\distill_leafAlex_1.h5")

    print(model.summary())



    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    history = model.fit_generator(MyTrainData(),
                                  steps_per_epoch=len(train_generator),
                                  epochs=epochs,
                                  validation_data=MyTestData(),
                                  validation_steps=len(test_generator),
                                  validation_freq=freq)
    model_alex.save(filepath=modelSave_path)
    print(history)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    leaf_data1 = {"epochs_range": epochs,
                  "acc": acc,
                  "val_acc": val_acc,
                  "loss": loss,
                  "val_loss": val_loss}

    # 数据线的保存和读取
    # Save
    f = open(graphSava_path, 'w')
    f.write(str(leaf_data1))
    f.close()

    # Load
    # 读取
    f = open(graphSava_path, 'r')
    a = f.read()
    dict_name = eval(a)
    f.close()
    print(dict_name)

if __name__ == "__main__":
    # DistillAlexNet()
    DistillVgg16()