import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

# =============tensorflow-gpu无法运行，用的下面代码解决===================
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# ====================================================================


model_path = r"D:\ML\model\ALL_chicken\chicken_02"
oneImage_path = r"C:\Users\伊尔安拉\Pictures\test\download.jpg"
SetImage_path = r"D:\ML\images\Chickens\chicken64"
file_path = r"D:\ML\images\newLeaf"

IMAGE_SIZE = (64, 64)

# 检测三十个图片
def Many_Test():
    # image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    image_generator = keras.preprocessing.image.ImageDataGenerator()
    image_data = image_generator.flow_from_directory(directory=SetImage_path,target_size=IMAGE_SIZE)

    class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
    print(class_names)
    class_names = np.array([key.title() for key, value in class_names])
    print(class_names)

    for i in class_names:
        print(i)
    for image_batch, label_batch in image_data:
        break
    print(image_batch.shape)

    for i in class_names:
        print("'{}',".format(i),end='')

    model = keras.models.load_model(filepath=model_path)

    # model = keras.Sequential([
    #      model,
    #      keras.layers.Softmax()
    # ])

    print(model.summary())
    predicted_batch = model.predict(image_batch)

    print(predicted_batch)

    predicted_id = np.argmax(predicted_batch,axis=-1)
    predicted_label_batch = class_names[predicted_id]

    # print(predicted_batch)

    label_id = np.argmax(label_batch, axis=-1)
    True_label = class_names[label_id]

    plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6,5,n+1)
        plt.imshow(image_batch[n]/255.0)
        color = 'green' if predicted_id[n] == label_id[n] else 'red'
        plt.title("pre{},Tr({})".format(predicted_label_batch[n].title(),True_label[n].title()), color=color)
        plt.axis('off')
    plt.suptitle("Model predictions (green: correct, red: incorrect)")
    plt.show()
# 检测一个图片
def alone_test():
    import PIL.Image as Image
    image = np.array(Image.open(oneImage_path)) / 255.0
    # 读取灰度图时用下面代码
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image,IMAGE_SIZE)
    print(np.array(image), np.array(image).shape)

    model = keras.models.load_model(filepath=model_path)


    # model = keras.Sequential([
    #     model,
        # keras.layers.Softmax()
    # ])
    print(model.summary())
    # class_names = ['中国流苏树','伯乐树','北美檫树','土沉香','大叶木莲','天竺桂',
    #                '弗吉尼亚木兰','榉树','水杉','水青树','猪血木','珙桐','红松','美国流苏树','连香树','鹅掌楸']
    class_names = ['Gong' 'Mu']

    # class_names = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]
    # print(class_names)

    # print(len(class_names))
    # print(class_names[22])
    prediet = model.predict(tf.expand_dims(image,0))
    print(oneImage_path)
    print("data:",prediet)
    prediet_id = np.argmax(prediet, axis=-1)
    print(prediet_id[0])
    True_label = class_names[prediet_id[0]]
    print(True_label)

    plt.imshow(image)
    plt.title(True_label)
    plt.show()
# 将源文件转化为tflite文件
def file_test():
    image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    image_data = image_generator.flow_from_directory(directory=file_path, target_size=IMAGE_SIZE)

    class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
    class_names = np.array([key.title() for key, value in class_names])


    for i in class_names:
        print("'{}',".format(i), end='')

    model = keras.models.load_model(filepath=model_path)

    model = keras.Sequential([
        model,
        keras.layers.Softmax()
    ])

    yes = 0
    no = 0
    number = 0
    for image_batch, label_batch in image_data:
        number+=1
        print(image_batch.shape)
        print(label_batch.shape)

        predicted_batch = model.predict(image_batch)
        predicted_id = np.argmax(predicted_batch, axis=-1)
        predicted_label_batch = class_names[predicted_id]
        label_id = np.argmax(label_batch, axis=-1)
        True_label = class_names[label_id]
        for n in range(32):
            if predicted_id[n] == label_id[n]:
                yes += 1
            else:
                no += 1
            print(n, ": ", end='')
            print("pre({}),Tr({})".format(predicted_label_batch[n].title(), True_label[n].title()))
        if number==1:
            break

    print("总数：",yes + no ,"正确：", yes,"=======","错误：", no)


def model_to_tflite(source_path, target_path):
    # model = keras.models.load_model(source_path)
    # model = keras.Sequential([
    #     model,
    #     keras.layers.Softmax()
    # ])
    # model.build(input_shape=(None,150,150,3))
    # print(model.summary())

    converter = tf.lite.TFLiteConverter.from_keras_model(source_path)
    tflite_model = converter.convert()

    # Save the model.
    with open(file=target_path, mode= 'wb') as f:
      f.write(tflite_model)


if __name__=="__main__":
    # alone_test()
    Many_Test()
    # file_test()

    # model_to_tflite(model_path, r"D:\ML\lite\leaf_5_8_c16.tflite")

