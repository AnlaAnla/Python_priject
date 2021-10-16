import tensorflow as tf
import tensorflow.keras as kears
import matplotlib.pyplot as plt

model_path = r"D:\ML\model\FaceVerification\mobilev2_faceV_03"
img_path = r""

def loadImage(path):
    tf.io.read_file(path)


model = kears.Sequential()
