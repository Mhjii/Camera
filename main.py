import cv2
import numpy as np

colorMin = (0,0,0)
colorMax = (0,0,0)

DeltaH = 10

DeltaS = 60
DeltaB = 60

DeltaR = 20
DeltaG = 20

lastImg = None
paused = False

laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# setup blob detector
# unused in example
detector = cv2.SimpleBlobDetector_create(params)


## on click function to caputre mouse events##
# takes in an event, the location of the event,
# any flags and the image that was clicked


def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

def onclick(event, x, y,flags,frame):
    # importing globals
    global colorMax, colorMin,lastImg,paused
    #Left mouse click selects color to mask with
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        color = frame[y,x]
        print('you just clicked me at' + str(refPt))
        print('the color here is' + str(color))
        colorMax = color + (DeltaH,DeltaS,DeltaB)
        print('New color max ' + str(colorMax))
        colorMin = color - (DeltaH, DeltaS, DeltaB)
        print('New color min ' + str(colorMin))

    # right click pauses the simulation
    if event == cv2.EVENT_RBUTTONDOWN:
        lastImg = frame
        paused = not paused


cv2.namedWindow("image")

## this is the look that allows the webcam to playback video
def show_webcam(mirror=False):
    global paused,lastImg
    # setting up our video capture
    cam = cv2.VideoCapture(0)
    # keep doing this forever
    while True:
        # check if paused
        if not paused:
            # if not get a new image
            ret_val, img = cam.read()
            # if the camera needs to be flipped
            # do so now
            if mirror:
                img = cv2.flip(img, 1)
        # if were paused keep displaying the last image
        else:

            img = lastImg
        # convert the image to HSV encoding
        # this allows similar colors to be more easily selected
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
        # generate a binary mask between our min and max colors
        mask = cv2.inRange(hsv,colorMin,colorMax)
        # generata a result
        res = cv2.bitwise_and(img,img,mask= mask)
        # show the mask
        cv2.imshow('mask', mask)

        convoleOutput = convolve(mask, laplacian)
        cvOutput = cv2.filter2D(mask, -1, laplacian)

        cv2.imshow('Conv: Edges',cvOutput)

        ## commented out to disable blob detection
        #keypoints = detector.detect(res)
        #blobImg = cv2.drawKeypoints(img,keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #cont = cv2.findContours(res,cv2.RETR_FLOODFILL,cv2.CHAIN_APPROX_SIMPLE)

        #contours = cv2.drawContours(img,[cont],-1,(0,255,0),2)
        # show the original image
        cv2.imshow('image', img)
        # setup the callbac to allow the clicks to work
        cv2.setMouseCallback("image", onclick, hsv)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        # set the last image to the current image
        lastImg = img
    # on exit close all windows
    cv2.destroyAllWindows()

# call the main loop
def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()