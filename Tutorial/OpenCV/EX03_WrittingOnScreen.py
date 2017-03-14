import numpy as np
import cv2

def writePoint(img_param, center_param):
    cv2.circle(img = img_param, center = center_param, radius = 5,
         color = (0,255,0), thickness = 10)
    cv2.putText(img = img_param, text = str(center_param),
     org = (center_param[0] - 20, center_param[1] - 20), fontFace = cv2.FONT_HERSHEY_PLAIN,
      fontScale = 1, color = (255,255,255), thickness = 3)

def main():

    print 'Press \'q\' to Exit'

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        writePoint(img_param = gray, center_param = (300, 300))
        writePoint(img_param = gray, center_param = (300, 500))
        writePoint(img_param = gray, center_param = (300, 700))
        writePoint(img_param = gray, center_param = (300, 900))
        writePoint(img_param = gray, center_param = (100, 300))
        writePoint(img_param = gray, center_param = (500, 300))
        writePoint(img_param = gray, center_param = (700, 300))
        writePoint(img_param = gray, center_param = (900, 300))

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()