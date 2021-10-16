from OpenCVTest.myTool.tool import loadImage
import cv2
import os
import sys
import numpy as np

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceCascade = cv2.CascadeClassifier(r"D:\Anaconda\envs\tensorflow-gpu\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")



while capture.isOpened():
    retval, frame = capture.read()
    faces = faceCascade.detectMultiScale(frame, 1.15)
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w] = [255, 155, 0]
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
    if retval == True:
        frame = cv2.Canny(frame, 120, 250)
        cv2.imshow("frame", frame)

    key = cv2.waitKey(50)
    if key == 32:
        cv2.waitKey(0)
        continue
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
