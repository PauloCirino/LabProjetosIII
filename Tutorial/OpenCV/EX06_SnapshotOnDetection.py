# import the necessary packages
import cv2
import time
import sys

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


def print_mouse_position_center(frame):
    global mouse_current_position_centert
    cv2.putText(img = frame, text = '(x, y) = ' + str(mouse_current_position_centert),
                org = (5, 20),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1, color = (255,255,255), thickness = 1)


def write_point(frame, center_param, color_param = (0, 255, 0)):
    cv2.circle(img = frame, center = center_param, radius = 5,
               color = color_param, thickness = 10)

    cv2.putText(img = frame, text = str(center_param),
                org = (center_param[0] - 20, center_param[1] - 20),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1, color = color_param, thickness = 1)

 
def draw_click(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        global mouse_current_position_centert
        mouse_current_position_centert = (x, y)

    if event == cv2.EVENT_LBUTTONUP:
        global click_down_point_center
        print '(x, y) = ' + str(click_down_point_center)
        click_down_point_center = (x, y)



def main(cascPath):

    print 'Click somewhere on the screen'
    print 'Press \'q\' to Exit'

    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)

    video_capture.set(3, 480)
    video_capture.set(4, 640)

    last_pic_time = time.time()

    last_time = time.time()
    frame_count = 0
    frame_rate = 0 

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_click)
	 
	# keep looping until the 'q' key is pressed
    while True:
        ret, frame = video_capture.read()

        global click_down_point_center
        write_point(frame = frame, center_param = click_down_point_center, color_param = (0, 255, 0))
        
        print_mouse_position_center(frame = frame)

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

            if (time.time() - last_pic_time) > 2 :
                print 'Saving Snapshot'
                last_pic_time = time.time()
                pic_name = './snapShots/face_' + str(last_pic_time) + '.png'
                frame2 = gray [range(y, y + h), :] [:, range(x, x + w)] 
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
    main(cascPath)
