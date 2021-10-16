import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 32
IMAGE_SHAPE = (256, 256)
EPOCHES = 30

src1Dir = r"D:\ML\images\BrighteningTrain\low"
dst1Dir = r"D:\ML\images\BrighteningTrain\high"
src2Dir = r"D:\ML\images\test\Littleface\enhence\low"
dst2Dir = r"D:\ML\images\test\Littleface\enhence\high"

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SHAPE)
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

train_paths = []
label_paths = []

# for name in os.listdir(src1Dir):
#     imgPath = src1Dir + '/' + name
#     labelPath = dst1Dir + '/' + name
#
#     train_paths.append(imgPath)
#     label_paths.append(labelPath)

for name in os.listdir(src2Dir):
    imgPath = src2Dir + '/' + name
    labelPath = dst2Dir + '/' + name

    train_paths.append(imgPath)
    label_paths.append(labelPath)

image_count = len(train_paths)
print("img number:", image_count)


train_path_ds = tf.data.Dataset.from_tensor_slices(train_paths)
label_path_ds = tf.data.Dataset.from_tensor_slices(label_paths)

image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
label_ds = label_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

# 显示部分图片
# for img, labelImg in image_label_ds:
#     plt.subplot(121)
#     plt.imshow(img)
#     plt.subplot(122)
#     plt.imshow(labelImg)
#     plt.show()

# ds = image_label_ds.apply(
#   tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds = ds.batch(BATCH_SIZE)
# ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds = image_label_ds.repeat()

def GenerateInputs(ds):
    for img, labelImg in ds:
        yield (tf.expand_dims(img, 0), tf.expand_dims(labelImg, 0))

# Input_Sample = Input(shape= IMAGE_SHAPE +(3,))
# Output_ = InstantiateModel(Input_Sample)
# Model_Enhancer = keras.Model(inputs=Input_Sample, outputs=Output_)

Model_Enhancer = keras.models.load_model(r"D:\ML\model\My_test\lowLight_sceneryEnhancer_06")
Model_Enhancer.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                       loss='mean_squared_error',
                       metrics=["accuracy"])
Model_Enhancer.summary()

# 训练增强器
Model_Enhancer.fit_generator(generator = GenerateInputs(ds),
                    steps_per_epoch = (image_count + BATCH_SIZE - 1) // BATCH_SIZE,
                    epochs = EPOCHES)

Model_Enhancer.save(r"D:\ML\model\My_test\lowLight_humanEnhancer_01")
