import numpy as np
import tensorflow as tf
import cv2
import tensorflow.keras as keras
import PIL.Image as Image

# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path=r"D:\Android\AndroidStudioProjects\android\models\src\main\assets\mobilenet_v1_1.0_224.tflite")
# interpreter.allocate_tensors()
#
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# print(input_details)
# print(output_details)



#
# ===================================================================================================
# 测试tflite模型

# # Load TFLite model and allocate tensors. interpreter = tf.contrib.lite.Interpreter(model_path="converted_model.tflite")
# interpreter = tf.lite.Interpreter(model_path=r"D:\ML\lite\newLeaf04.tflite")
# interpreter.allocate_tensors()
#
# # Get input and output tensors
#
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# Test model on random input data

# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape),dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)



# IMAGE_SIZE = (224, 224)
# img = np.array(Image.open(r"C:\Users\伊尔安拉\Pictures\Saved Pictures\12443929_141820962000_2.jpg"))/255.0
# img = tf.image.resize(img ,IMAGE_SIZE)
#
# print(img.shape)
#
# interpreter.set_tensor(0, tf.expand_dims(img,0))
#
# interpreter.invoke()
#
# output_data = interpreter.get_tensor(output_details[0]['index'])
#
# print(output_data)


# ===========================================================================







model = keras.models.load_model(r"D:\ML\model\ALL_chicken\chicken_02")

# model = keras.Sequential([
#     keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(150, 150, 3)),
#     s_model
# ])
model.build()

print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(file=r"D:\ML\lite\heafADDRescal_Checken_02.tflite", mode='wb') as f:
    f.write(tflite_model)
print('end')