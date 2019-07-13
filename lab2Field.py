import cv2

def show_webcam(mirror=False):
    global paused,lastImg,colorMax,colorMin,color,frameCount
    # setting up our video capture
    cam = cv2.VideoCapture(0)
    # keep doing this forever
    ret_val, img = cam.read()
    # if the camera needs to be flipped
    # do so now
    if mirror:
        img = cv2.flip(img, 1)


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()