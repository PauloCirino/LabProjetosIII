import numpy as np
import cv2

def main():

    print 'Press \'q\' to Exit'

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print str(gray.shape)

        cv2.circle(img = gray, center = (450,300), radius = 70,
         color = (0,0,255), thickness = 15
         )
        cv2.circle(img = gray, center = (750,300), radius = 70,
         color = (0,0,255), thickness = 15
         )
        cv2.ellipse(img = gray, center = (640, 600), axes = (100,50),
         angle = 0, startAngle = 0, endAngle = 180, color = (255, 0, 0),
          thickness = 15
         )
        cv2.rectangle(img = gray, pt1 = (350, 50), pt2 = (900, 700),
         color = (125, 125, 60), thickness = 20)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()