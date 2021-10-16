import tensorflow.keras as keras
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from keras_flops import get_flops
# build model

# build model
# inp = Input((32, 32, 1))
# x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Flatten()(x)
# x = Dense(128, activation="relu")(x)
# out = Dense(10, activation="softmax")(x)
# model = Model(inp, out)

# # build model
# build model
# inp = Input((32, 32, 1))

IMAGE_SIZE = (150, 150)
classes = 16

model_alex = keras.Sequential(name='AlexNet_03')
model_alex.add(keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)))
model_alex.add(
    Conv2D(64, (11, 11), strides=(4, 4), padding='valid', activation='relu', kernel_initializer='uniform'))
model_alex.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model_alex.add(
    Conv2D(150, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model_alex.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# model_alex.add(
#     Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model_alex.add(
#     Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
# model_alex.add(
#     Conv2D(150, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model_alex.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model_alex.add(Flatten())
model_alex.add(Dense(256, activation='relu'))
model_alex.add(Dense(128, activation='relu'))
model_alex.add(Dense(classes, activation='softmax'))

print(model_alex.summary())

flops = get_flops(model_alex, batch_size=1)
print(f"flops: {flops / 10 ** 9:.03} g")
