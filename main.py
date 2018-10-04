import cv2
import numpy as np
import pylivestream as stream

colorMin = (0,0,0)
colorMax = (0,0,0)

DeltaH = 10

DeltaS = 50
DeltaB = 75
DeltaR = 20
DeltaG = 20

lastImg = None
paused = False

kernel = np.ones((21,21),np.float32)/441


laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

topR = np.array((
	[-4, 1, 1],
	[1, 0, 0],
	[1, 0, 0]), dtype="int")

topL = np.array((
	[1, 1, -4],
	[0, 0, 1],
	[0, 0, 1]), dtype="int")

botR = np.array((
	[1, 0, 0],
	[1, 0, 0],
	[-4, 1, 1]), dtype="int")

botL = np.array((
	[0, 0, 1],
	[0, 0, 1],
	[1, 1, -4]), dtype="int")

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
params.filterByConvexity = True
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

def onclick(event, x, y,flags,frame = None):
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
        if paused:
            print('paused')
        else:
            print('unpaused')

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

            blur = cv2.filter2D(img, -1, kernel)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS_FULL)
            mask = cv2.inRange(hsv, colorMin, colorMax)
            # generata a result

            res = cv2.bitwise_and(img, img, mask=mask)
            # show the mask
            cv2.imshow('mask', mask)
            cv2.imshow('Bitwise', res)
            # convoleOutput = convolve(mask, laplacian)
            laplac = cv2.filter2D(mask, -1, laplacian)
            # topRight = cv2.filter2D(mask, -1, topR)
            # bottomLeft = cv2.filter2D(mask, -1, botL)
            # bottomRight = cv2.filter2D(mask, -1, botR)

            cv2.imshow('Conv: Edges', laplac)
            ret, thresh = cv2.threshold(mask, 200, 255, 3)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            try:
                maxArea = 0
                # for cont in contours:
                #     if cv2.contourArea(cont) > maxArea:
                #         cnt = cont
                cnt = contours[0]
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                cv2.drawContours(img, contours, 0, (0, 255, 255), 2)
                points = np.array(box)
                center = np.sum(points, 0) / 4
                print(center[0])

                cv2.drawMarker(img, (box[0][0], box[0][1]), (255, 0, 0), cv2.MARKER_CROSS)
                cv2.drawMarker(img, (box[1][0], box[1][1]), (255, 0, 0), cv2.MARKER_CROSS)
                cv2.drawMarker(img, (box[2][0], box[2][1]), (255, 0, 0), cv2.MARKER_CROSS)
                cv2.drawMarker(img, (box[3][0], box[3][1]), (255, 0, 0), cv2.MARKER_CROSS)
                cv2.drawMarker(img, (int(center[0]), int(center[1])), (255, 165, 0), cv2.MARKER_TRIANGLE_UP)

            except:
                cv2.drawMarker(img, (0, 0), (255, 165, 0), cv2.MARKER_TRIANGLE_UP)

            # leftmost = tuple(laplac[laplac[:, :, 0].argmin()][0])
            # rightmost = tuple(laplac[laplac[:, :, 0].argmax()][0])
            # topmost = tuple(laplac[laplac[:, :, 1].argmin()][0])
            # bottommost = tuple(laplac[laplac[:, :, 1].argmax()][0])

            ## commented out to disable blob detection
            # keypoints = detector.detect(img)
            # blobImg = cv2.drawKeypoints(img,keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # img = cv2.circle(img, leftmost, 5, (0,0,255), -1)
            # img = cv2.circle(img, rightmost, 5, (0, 0, 255), -1)
            # img = cv2.circle(img, topmost, 5, (0, 0, 255), -1)
            # img = cv2.circle(img, bottommost, 5, (0, 0, 255), -1)

            # cont = cv2.findContours(res,cv2.RETR_FLOODFILL,cv2.CHAIN_APPROX_SIMPLE)

            # contours = cv2.drawContours(img,[cont],-1,(0,255,0),2)
            # show the original image
            cv2.imshow('image', img)
            # setup the callbac to allow the clicks to work
            cv2.setMouseCallback("image", onclick, hsv)
        # if were paused keep displaying the last image
        else:
            img = lastImg
            cv2.imshow('image', img)
            cv2.setMouseCallback("image", onclick)
        # convert the image to HSV encoding
        # this allows similar colors to be more easily selected

        # generate a binary mask between our min and max colors

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