import cv2
import numpy as np
import matplotlib.pyplot as plt


def loadImage(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, -1)
    return img