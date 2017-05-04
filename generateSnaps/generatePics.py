# import the necessary packages
import cv2
import time
import sys
import os

time_between_pics = 0.25
VAR_FOLGA_X = 25
VAR_FOLGA_Y = 60
click_up_point_center = (-100, -100)
click_down_point_center = (-100, -100)

mouse_current_position_centert = (0, 0)


def print_fps(frame, last_time, frame_count, frame_rate):
        frame_count += 1
        if (time.time() - last_time) > 1.0:
            frame_rate = frame_count
            last_time = time.time()
            frame_count = 1

        cv2.putText(img = frame, text = str(frame_rate) + ' FPS',
                    org = (5, 10),
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 1, color = (255,255,255), thickness = 1)

        return last_time, frame_count, frame_rate


def get_face_frame(original_frame, x, y, w, h, var_folga_x, var_folga_y):
    print original_frame.shape

    newYMax = original_frame.shape[0]
    newXMax = original_frame.shape[1]
    newXMin = 0
    newYMin = 0

    if x - var_folga_x > 0 :
        newXMin = x - var_folga_x
    if x + w + var_folga_x < newXMax :
        newXMax = x + w

    if y - var_folga_y > 0 :
        newYMin = y - var_folga_y
    if y + h + var_folga_y < newYMax :
        newYMax = y + h 

    frame = original_frame[range(newYMin, newYMax), :] [:, range(newXMin, newXMax)] 
    return frame


def main(cascPath, snapDir, personName):

    print 'Press \'q\' to Exit'

    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)

    video_capture.set(3, 480)
    video_capture.set(4, 640)

    last_pic_time = time.time()
    last_pic_num = 0

    last_time = time.time()
    frame_count = 0
    frame_rate = 0 

    cv2.namedWindow('image')
	 
	# keep looping until the 'q' key is pressed
    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (50, 50),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if (time.time() - last_pic_time) > time_between_pics :
                print 'Saving Snapshot'
                last_pic_time = time.time()
                last_pic_num += 1
                pic_name = os.path.join(snapDir, personName + str(last_pic_num) + '.png')
                frame2 = get_face_frame(gray, x, y, w, h, VAR_FOLGA_X, VAR_FOLGA_Y)
                print 'frame2 shape = ' + str(frame2.shape)
                cv2.imwrite(pic_name, frame2)

		# prints frames per second
        last_time, frame_count, frame_rate = print_fps(frame, last_time, frame_count, frame_rate)

		# display the image and wait for a keypress
        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    cascPath = sys.argv[1]
    snapDir = sys.argv[2] 
    personName = sys.argv[3]

    try:
        os.stat(snapDir)
    except:
        os.mkdir(snapDir) 

    snapDir = os.path.join(snapDir, personName)
    try:
        os.stat(snapDir)
    except:
        os.mkdir(snapDir) 

    main(cascPath, snapDir, personName)
