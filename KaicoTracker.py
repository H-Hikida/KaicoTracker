from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
    parser.add_argument('--learningRate', type=float, help='Learning Rate for apply method', default=-1)
    parser.add_argument('--blurringSquare', type=int, help='Edge length of blurring square', default=10)
    parser.add_argument('--top', type=int, help='top position of frame', default=-1)
    parser.add_argument('--bottom', type=int, help='bottom position of frame', default=0)
    parser.add_argument('--left', type=int, help='left position of frame', default=0)
    parser.add_argument('--right', type=int, help='right position of frame', default=-1)
    args = parser.parse_args()
    if args.algo == 'KNN':
        backSub = cv.createBackgroundSubtractorKNN()
    else:
        backSub = cv.createBackgroundSubtractorMOG2()
    backSub.setDetectShadows(False)
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)
    
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        # cropping
        if args.top < 0:
            top = frame.shape[0]
        else:
            top = args.top
        if args.right < 0:
            right = frame.shape[1]
        else:
            right = args.right
        frameCropped = frame[args.bottom: top, args.left: right]

        # background subtraction
        fgMask = backSub.apply(frameCropped, learningRate=args.learningRate)
        # blurring foreground
        fgMaskBlur = cv.blur(fgMask, (args.blurringSquare, args.blurringSquare))
        # display blurred foreground
        cv.imshow('FG Mask blur', fgMaskBlur)

        # display current frame #
        cv.rectangle(frameCropped, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frameCropped, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        # diplay frame with frame #
        cv.imshow('Frame', frameCropped)
        

        if capture.get(cv.CAP_PROP_POS_FRAMES) > 0:
            canny_output = cv.Canny(fgMaskBlur, 100, 100 * 2)
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # Draw contours
            drawing = np.zeros((frameCropped.shape[0], frameCropped.shape[1], 3), dtype=np.uint8)
            approxContours = []
            hullContours = []
            for i in contours:
                # Contour Approximation
                epsilon = 0.01*cv.arcLength(i, True)
                approx = cv.approxPolyDP(i, epsilon, True)
                approxContours.append(approx)
                # Minimum Enclosing Circle
                (x,y),radius = cv.minEnclosingCircle(i)
                center = (int(x),int(y))
                radius = int(radius)
                cv.circle(drawing,center,radius,(0,0,255),2)
            for i in range(len(approxContours)):
                cv.drawContours(drawing, approxContours, i, (255, 0, 0), 2, cv.LINE_8, hierarchy, 0)
            cv.imshow('Contours', drawing)
        
        
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
