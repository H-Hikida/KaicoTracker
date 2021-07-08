from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from matplotlib import pyplot as plt
import seaborn as sns
import cv2 as cv
import argparse
import tqdm
import itertools


def defineBorders(df):
    borders = {}
    for i in ['XM', 'YM']:
        Slices = np.linspace(1, int(df[i].max()), num=int(df[i].max()))
        kernel = stats.gaussian_kde(df[i])
        estimates = kernel(Slices)
        borders[i] = [0] + signal.argrelmin(estimates)[0].tolist() + [df[i].max()]
        print(i, borders[i])
        plt.scatter(Slices, estimates, s=0.1, c="gray")
        plt.savefig('{}_{}_kde.png'.format(args.prefix, i), format="png", dpi=200)
        plt.close()
    return borders['XM'], borders['YM']


def mergePoints(df, total):
    outDfs = []
    for i in range(1,total):
        points = df[df.Slice == i]
        if len(points) > 0:
            outX = int(points.XM.mean())
            outY = int(points.YM.mean())
            outDfs.append(pd.DataFrame({'Slice': i, 'XM': outX, 'YM': outY, 'file': points.file.tolist()[0]}, index=[i]))
    return pd.concat(outDfs, sort=False)


def DistantCalc(data, period):
    times = []
    for i in range(1, period+1):
        time = data[data.Slice == i]
        if len(time) == 1:
            times.append(time)
        elif (len(time) == 0) & (i > 1):
            times.append(times[(i-1)-1])
        elif (len(time) == 0) & (i == 1):
            df_first = data[data.Slice == (i+1)]
            while(len(df_first) == 0):
                i += 1
                df_first = data[data.Slice == (i+1)]
            times.append(df_first)
    data1 = pd.concat(times, sort=False)
    data1.Slice = range(1, period+1)
    dist = np.sqrt(\
                   np.power(data1.XM.values[0:len(data1)-1]-data1.XM.values[1:len(data1)], 2)\
                   + np.power(data1.YM.values[0:len(data1)-1]-data1.YM.values[1:len(data1)], 2)\
                  )
    data1["distance"] = [0] + list(dist)
    accdist = np.cumsum(data1.distance)
    data1["accumlated"] = accdist
    return (data1, list(accdist)[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, nargs='+', help='Path to a video or a sequence of image.', default=['vtest.avi'])
    parser.add_argument('--prefix', type=str, help='Prefix of path to output files.', default='stdout')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
    parser.add_argument('--learningRate', type=float, help='Learning Rate for apply method', default=-1)
    parser.add_argument('--blurringSquare', type=int, help='Edge length of blurring square', default=10)
    parser.add_argument('--live', action='store_true', help='Display processing movie in live, default=False')
    parser.add_argument('--noPoint', action='store_false', help='Pointing, default=True')
    parser.add_argument('--fps', type=int, help='output FPS', default=-1)
    parser.add_argument('--top', type=int, help='top position of frame', default=-1)
    parser.add_argument('--bottom', type=int, help='bottom position of frame', default=0)
    parser.add_argument('--left', type=int, help='left position of frame', default=0)
    parser.add_argument('--right', type=int, help='right position of frame', default=-1)
    args = parser.parse_args()

    #----------------------# Tracking #----------------------#
    if args.algo == 'KNN':
        backSub = cv.createBackgroundSubtractorKNN()
    else:
        backSub = cv.createBackgroundSubtractorMOG2()
    backSub.setDetectShadows(False)
    df_captures = []
    startSlice = 0
    for inCap in args.input:
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(inCap))
        fcount = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened():
            print('Unable to open: ' + inCap)
            exit(0)
        if args.top < 0:
            top = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        else:
            top = args.top
        if args.right < 0:
            right = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        else:
            right = args.right
        
        dfs =[]
        print("Start Tracking...")
        with tqdm.tqdm(total=fcount) as pbar:
            while True:
                ret, frame = capture.read()
                if frame is None:
                    break
                pbar.update(1)
                frame_num = int(capture.get(cv.CAP_PROP_POS_FRAMES)) + startSlice

                # cropping & background subtraction & blurring foreground
                frameCropped = frame[args.bottom: top, args.left: right]
                fgMask = backSub.apply(frameCropped, learningRate=args.learningRate)
                fgMaskBlur = cv.blur(fgMask, (args.blurringSquare, args.blurringSquare))

                # display current frame #
                cv.rectangle(frameCropped, (10, 2), (100,20), (255,255,255), -1)
                cv.putText(frameCropped, str(frame_num), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

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
                    # make DataFrame row
                    d = {'Slice': [int(frame_num)], 'XM': [int(x)], "YM": [int(y)], 'radius': [int(radius)]}
                    dfs.append(pd.DataFrame(data=d, index=[0]))
                for i in range(len(approxContours)):
                    cv.drawContours(drawing, approxContours, i, (255, 0, 0), 2, cv.LINE_8, hierarchy, 0)
                    
                # diplay movie if --live flag is true
                if args.live:
                    cv.imshow('Frame', frameCropped)
                    cv.imshow('FG Mask blur', fgMaskBlur)
                    cv.imshow('Contours', drawing)
                
                keyboard = cv.waitKey(30)
                if keyboard == 'q' or keyboard == 27:
                    break
            
        df_temp = pd.concat(dfs, sort=False)
        df_temp["file"] = inCap
        df_captures.append(df_temp)
        cv.destroyAllWindows()
        startSlice += fcount
    df = pd.concat(df_captures, sort=False, ignore_index=True)
    df.to_csv('{}.txt'.format(args.prefix), sep='\t', index=False)
    totalSlice = startSlice

    #----------------------# Analyzing #----------------------#
    print("Start border determination...")
    xborder, yborder = defineBorders(df)
    merged_list = []
    print("Start analyzing...")
    # Distant calculation
    with open('{}_totalDist.txt'.format(args.prefix), "w") as outDist:
        with tqdm.tqdm(total=(len(xborder)-1)*(len(yborder)-1)) as pbar:
            for i, j in itertools.product(range(len(xborder)-1), range(len(yborder)-1)):
                cut_bottom = yborder[j]
                cut_top = yborder[j+1]
                cut_left = xborder[i]
                cut_right = xborder[i+1]
                area_id = '{}_{}_{}_{}'.format(cut_left, cut_right, cut_bottom, cut_top)
                temp = df[(df.XM > cut_left) & (df.XM < cut_right) & (df.YM > cut_bottom) & (df.YM < cut_top)]
                merged = mergePoints(temp, totalSlice)
                merged['area'] = area_id
                merged_list.append(merged)
                dfDist, totalDist = DistantCalc(merged, totalSlice)
                outDist.write('\t'.join([area_id, str(totalDist)])+'\n')
                plt.scatter('Slice', 'distance', data=dfDist, s=0.1, alpha=0.5, c="gray")
                sns.despine()
                plt.savefig('{}_distance_{}.png'.format(args.prefix, area_id), format="png", dpi=200)
                plt.close()
                pbar.update(1)
    dfMerged = pd.concat(merged_list, axis=0, sort=False, ignore_index=True)
    dfMerged.to_csv('{}_positions.txt'.format(args.prefix), sep='\t', index=False)
    plt.scatter(dfMerged.XM, dfMerged.YM, c="gray", s=0.1)
    plt.hlines(yborder, colors="lightgray", linewidth=1, xmin=min(xborder), xmax=max(xborder))
    plt.vlines(xborder, colors="lightgray", linewidth=1, ymin=min(yborder), ymax=max(yborder))
    plt.savefig('{}_positions.png'.format(args.prefix), format="png", dpi=200)
    plt.close()

    #----------------------# Pointing #----------------------#
    # Video setting
    if args.noPoint:
        print("Start pointing...")
        for inCap in args.input:
            capture = cv.VideoCapture(cv.samples.findFileOrKeep(inCap))
            fcount = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
            if args.fps < 0:
                fps = capture.get(cv.CAP_PROP_FPS)
            else:
                fps = args.fps 
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            outVideo = cv.VideoWriter("pointed_{}.mp4".format(inCap), fourcc, fps, (right-args.left, top-args.bottom))
            with tqdm.tqdm(total=fcount) as pbar:
                while True:
                    ret, frame = capture.read()
                    if frame is None:
                        break
                    pbar.update(1)
                    frame_num = int(capture.get(cv.CAP_PROP_POS_FRAMES))
                    frameCropped = frame[args.bottom: top, args.left: right]
                    temp = dfMerged[(dfMerged['Slice'] == frame_num) & (dfMerged['file'] == inCap)]
                    if len(temp) > 0:
                        for i in temp.index:
                            x = int(temp.loc[i, 'XM'])
                            y = int(temp.loc[i, 'YM'])
                            frameCropped = cv.circle(frameCropped, (x, y), 5, color=(0, 0, 255), thickness=-1)
                    im_gray = cv.cvtColor(frameCropped, cv.COLOR_BGR2GRAY)
                    if args.live:
                        cv.imshow('outVideo', im_gray)
                    outVideo.write(im_gray)

                    keyboard = cv.waitKey(30)
                    if keyboard == 'q' or keyboard == 27:
                        break
