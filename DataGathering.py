# Imports
import os
import time
import uuid
import cv2

# Constants
IMAGES_PATH = os.path.join("data", "images")
NUMBER_IMAGES = 30

# Gather data
cap = cv2.VideoCapture(0)
for imgnum in range(NUMBER_IMAGES):
    print(f"Collecting image {imgnum}")
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH, f"{str(uuid.uuid1())}.jpg")
    cv2.imwrite(imgname, frame)
    cv2.imshow("frame", frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
