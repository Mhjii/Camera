import cv2
import numpy as np
from firebase import dataBase
import time

colorMin = (0,0,0)
colorMax = (0,0,0)

DeltaH = 10

DeltaS = 50
DeltaB = 100
DeltaR = 20
DeltaG = 20

minArea = 300       # threshold for blob to be detected
AbsMaxArea = 180000 # cuttoff point for thrust, determines what blob size signifies a landing

frameCount = 0      # global frame counter for intermittent events

color = np.zeros(3)

lastImg = None
paused = False

tick = time.time()  # timing objects to calculate fps
tock = time.time()

kernel = np.ones((21,21),np.float32)/441 # gaussian blur kernel

## on click function to caputre mouse events##
# takes in an event, the location of the event,
# any flags and the image that was clicked

#firebase = dataBase()
#firebase.init()

def onclick(event, x, y,flags,frame = None):
    # importing globals
    global colorMax, colorMin,lastImg,paused,color
    #Left mouse click selects color to mask with
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        color = frame[y,x]
        print(refPt[0])
        print('Just clicked at: \t' + str('[{:} {:}]'.format(refPt[0][0],refPt[0][1])))
        print('Selected Color: \t' + str(color))
        colorMax = color + (DeltaH,DeltaS,DeltaB)
        for i,val in enumerate(colorMax):
            if val > 255:
                colorMax[i] = 255
        print('New color max:\t\t' + str(colorMax))
        colorMin = color - (DeltaH, DeltaS, DeltaB)
        for i,val in enumerate(colorMin):

            if val < 0:
                colorMin[i] = 0
        print('New color min:\t\t' + str(colorMin))
        #firebase.storeColors(colorMin,color,colorMax)

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
    global paused,lastImg,colorMax,colorMin,color,frameCount
    global tick,tock
    # setting up our video capture
    cam = cv2.VideoCapture(0)
    # keep doing this forever
    while True:
        # check if paused
        if not paused:
            frameCount += 1
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

            #res = cv2.bitwise_and(img, img, mask=mask)
            # show the mask
            #cv2.imshow('mask', mask)
            #cv2.imshow('Bitwise', res)
            cv2.meanStdDev(img,color,None,mask)
            colorMaxNew = color.astype(int) + (DeltaH, DeltaS, DeltaB)
            colorMinNew = color.astype(int) - (DeltaH, DeltaS, DeltaB)

            # convoleOutput = convolve(mask, laplacian)
            #laplac = cv2.filter2D(mask, -1, laplacian)
            # topRight = cv2.filter2D(mask, -1, topR)
            # bottomLeft = cv2.filter2D(mask, -1, botL)
            # bottomRight = cv2.filter2D(mask, -1, botR)

            #cv2.imshow('Conv: Edges', laplac)
            ret, thresh = cv2.threshold(mask, 200, 255, 3)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            try:
                maxArea = minArea
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > maxArea:
                        maxArea = area
                        rect = cv2.minAreaRect(cnt)


                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                cv2.drawContours(img, contours, 0, (0, 255, 255), 2)
                points = np.array(box)
                center = np.sum(points, 0) / 4
                cv2.drawMarker(img, (box[0][0], box[0][1]), (255, 0, 0), cv2.MARKER_CROSS)
                cv2.drawMarker(img, (box[1][0], box[1][1]), (255, 0, 0), cv2.MARKER_CROSS)
                cv2.drawMarker(img, (box[2][0], box[2][1]), (255, 0, 0), cv2.MARKER_CROSS)
                cv2.drawMarker(img, (box[3][0], box[3][1]), (255, 0, 0), cv2.MARKER_CROSS)
                cv2.drawMarker(img, (int(center[0]), int(center[1])), (255, 165, 0), cv2.MARKER_TRIANGLE_UP)
                cv2.arrowedLine(img,(320,240),(int(center[0]), int(center[1])), (255, 100, 255),5)
                rectArea = boxArea(box)
                z = rectArea/307200
                vector = pix2vec(center,[320,240],z)


            except:
                cv2.drawMarker(img, (0, 0), (255, 165, 0), cv2.MARKER_TRIANGLE_UP)

            if frameCount == 30:
                tock = time.time()
                #firebase.storeCentroid(center)

                fTime = tock - tick
                FPS = frameCount/fTime

                print('Heading: [{:.4f},{:.4f},{:.4f}]\tFPS: {:.2f}'.format(vector[0],vector[1],vector[2],FPS))
                tick = time.time()
                frameCount = 0
            cv2.imshow('image', img)
            # setup the callbac to allow the clicks to work
            cv2.setMouseCallback("image", onclick, hsv)
            colorMax = colorMaxNew
            colorMin = colorMinNew
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

def pix2vec(point,center,blobRatio):
    ## returns the unit vector for the given point and center
    x = point[0]-center[0]
    y = center[1]-point[1]
    z = blobRatio

    m = np.sqrt(x*x+y*y+z*z)
    vx = x/m
    vy = y/m
    vz = z/m
    m = np.sqrt(vx * vx + vy * vy + vz * vz)
    return [vx, vy, vz]

def boxArea(box):
    ## returns the bounding area of a openCV Box2D object
    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[2]
    s1x = x1 - x2
    s1y = y1 - y2
    s2x = x1 - x3
    s2y = y1 - y3
    s1 = np.sqrt(s1x*s1x + s1y*s1y)
    s2 = np.sqrt(s2x*s2x + s2y*s2y)
    area = s1*s2
    return area

if __name__ == '__main__':
    main()