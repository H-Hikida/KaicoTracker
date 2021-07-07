from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import random as rng


def thresh_callback(img, val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(img, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        #color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, (255, 255, 255), 2, cv.LINE_8, hierarchy, 0)
    # Show in a window
    return drawing
    


if __name__ == '__main__':
    rng.seed(12345)
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    parser.add_argument('--learningRate', type=float, help='Learning Rate for apply method', default=-1)
    args = parser.parse_args()
    if args.algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()
    backSub.setDetectShadows(False)
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)
    
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        fgMask = backSub.apply(frame, learningRate=args.learningRate)
        
        
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        #src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #src_gray = cv.blur(src_gray, (3,3))
        #drawing = thresh_callback(frame, 100)
        #fgMask = backSub.apply(drawing, learningRate=args.learningRate)

        cv.imshow('Frame', frame)
        #cv.imshow('FG Mask', fgMask)
        fgMaskBlur = cv.blur(fgMask, (10, 10))
        cv.imshow('FG Mask blur', fgMaskBlur)

        if capture.get(cv.CAP_PROP_POS_FRAMES) > 0:
            canny_output = cv.Canny(fgMaskBlur, 100, 100 * 2)
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # Draw contours
            drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            approxContours = []
            hullContours = []
            for i in contours:
                # Contour Approximation
                epsilon = 0.01*cv.arcLength(i, True)
                approx = cv.approxPolyDP(i, epsilon, True)
                approxContours.append(approx)
                # Convex Hull
                #hull = cv.convexHull(i)
                #hullContours.append(hull)
                # Minimum Enclosing Circle
                (x,y),radius = cv.minEnclosingCircle(i)
                center = (int(x),int(y))
                radius = int(radius)
                cv.circle(drawing,center,radius,(0,0,255),2)
            for i in range(len(approxContours)):
                cv.drawContours(drawing, approxContours, i, (255, 0, 0), 2, cv.LINE_8, hierarchy, 0)
                #cv.drawContours(drawing, hullContours, i, (0, 0, 255), 2, cv.LINE_8, hierarchy, 0)
            cv.imshow('Contours', drawing)
        
        
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
