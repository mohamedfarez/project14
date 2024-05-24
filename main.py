# import libraries

import cv2

"""
Initializes the face and eye detectors using pre-trained Haar Cascade classifiers.

The `face_detector` and `eye_detector` variables are used to detect faces and eyes in images or video frames using OpenCV's Haar Cascade-based object detection algorithms.

The `haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` files contain the pre-trained Haar Cascade classifiers for detecting frontal faces and eyes, respectively. These files are loaded into the `CascadeClassifier` objects to enable the detection functionality.
"""
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

cam = cv2.VideoCapture(0)

ret = True

while ret:
    ret, frame = cam.read()

    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_points:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
        
            face = frame[y:y+h,x:x+w]
            eyes = eye_detector.detectMultiScale(face,1.3,5)
            for (x, y, w, h) in eyes:
                cv2.rectangle(face, (x, y), (x+w, y+h), (155, 0, 120), 2)

        cv2.imshow('Live feed', frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()