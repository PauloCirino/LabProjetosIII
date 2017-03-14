# TO Run
# $python EX04_TrackingFaces.py ./../../data/csv2/haarcascade_frontalface_default.xml

import numpy as np
import cv2
import sys
import time

def main(cascPath):
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)

    video_capture.set(3, 360)
    video_capture.set(4, 640)


    last_time = time.time()
    frames_count = 0
    frame_rate = 0 

    print "Press \'q\' on the video window to exit!"

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        frames_count += 1
        if (time.time() - last_time) > 1.0:
            frame_rate = frames_count
            last_time = time.time()
            frames_count = 1

        cv2.putText(img = frame, text = str(frame_rate) + ' FPS',
                    org = (5, 10),
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 1, color = (255,255,255), thickness = 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cascPath = sys.argv[1]
    main(cascPath)