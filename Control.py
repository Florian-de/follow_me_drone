import numpy as np
from djitellopy import tello
from time import sleep
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

width, height = 120, 120
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
p_error = 0

drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.streamon()
drone.takeoff()
drone.send_rc_control(0,0,25,0)
sleep(3)
drone.send_rc_control(0,0,0,0)
facetracker = load_model("facetracker")

def findFace(img):
    """
    Detecs faces in given image
    :param img: image in which we want to detect faces
    :return: img, center and area of the largest face in the image
    """
    """face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.2, 8)
    """
    face = facetracker.predict(img)
    if face[0][0][0] < 0.4:
        return img, [[0, 0], 0]
    x_1, y_1 = tuple(np.multiply(face[1][0][:2], [120, 120]).astype(int))
    x_2, y_2 = tuple(np.multiply(face[1][0][2:], [120, 120]).astype(int))
    w = abs(x_1-x_2)
    h = abs(y_1-y_2)
    c_x = (x_1+x_2) // 2
    c_y = (y_1+y_2) // 2
    #for (x, y, w, h) in faces:
    cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (0,0,255), 2)
    area = w * h
    cv2.circle(img, (c_x, c_y), 5, (0,255, 0), cv2.FILLED)
    if area > 0:
        return img, [[c_x, c_y], area]


def trackFace(drone, info, width, pid, pError):
    """
    Calculates how the drone should move
    :param drone: which should be controled
    :param info: classification class
    :param width: of image
    :param pid: -
    :param pError: -
    :return: NOne
    """
    area = info[1]
    x, y = info[0]

    error = x - width//2
    speed = pid[0] * error + pid[1] * (error-pError)
    speed = int(np.clip(speed, -100, 100))
    # handle small false positives
    if area < 3000:
        speed = 0
    fb = 0

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    if area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area > 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0
    drone.send_rc_control(0, fb, 0, speed)
    return error

while True:
    img = drone.get_frame_read().frame
    #img = img[50:500, 50:500, :]
    #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(img, (120, 120))
    img, info = findFace(np.expand_dims(resized/255, 0))
    p_error = trackFace(drone, info, width, pid, p_error)
    #cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        drone.land()
        break
