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
from multiprocessing import Pool


def defineBorders(df):
    borders = {}
    for i in ['XM', 'YM']:
        Slices = np.linspace(1, int(df[i].max()), num=int(df[i].max()))
        kernel = stats.gaussian_kde(df[i])
        estimates = kernel(Slices)
        borders[i] = [0] + signal.argrelmin(estimates)[0].tolist() + [df[i].max()]
        print(i, borders[i])
        plt.scatter(Slices, estimates, s=0.1, c="gray")
        plt.savefig('{}_{}_kde.{}'.format(args.prefix, i, args.format), format=args.format, dpi=200)
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


def DistantCalc(data, period, id):
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
    data1["id"] = id
    plt.figure(figsize=(6,2))
    plt.scatter('Slice', 'distance', data=data1, s=0.1, alpha=0.5, c="gray")
    plt.ylim(-1,20)
    sns.despine()
    plt.savefig('{}_distance_{}.{}'.format(args.prefix, id, args.format), format=args.format, dpi=200)
    plt.close()
    sns.lineplot('Slice', 'accumlated', data=data1, linewidth=1, color='gray')
    sns.despine()
    plt.savefig('{}_accdist_{}.{}'.format(args.prefix, id, args.format), format=args.format, dpi=200)
    plt.close()
    active = data1[data1.distance>0]
    plt.figure(figsize=(3,1))
    sns.distplot(active.distance)
    plt.savefig('{}_speed_distplot_{}.{}'.format(args.prefix, id, args.format), format=args.format, dpi=200)
    plt.close()
    return (data1, list(accdist)[-1], active.distance.median())


def CalcAreaChange(df, totalSlice, id, length=-1):
    _, axs = plt.subplots(nrows=1, ncols=args.segment+1, figsize=(args.segment+1, 1))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    Xcen = df.XM.mean()
    Ycen = df.YM.mean()
    if length < 0:
        Xstd = df.XM.std()
        Ystd = df.YM.std()
        length = max(Xstd, Ystd) * 3
    for k in range(0, args.segment+1):
        trange = totalSlice // args.segment
        part = df[(df.Slice > trange * k) & (df.Slice <= trange * (k+1))]
        axs[k].set_xlim(Xcen-length, Xcen+length)
        axs[k].set_ylim(Ycen-length, Ycen+length)
        axs[k].scatter(part.XM, part.YM, s=0.1, alpha=0.5, c='gray')
        axs[k].set_xticklabels([])
        axs[k].set_yticklabels([])
        axs[k].set_xticks([])
        axs[k].set_yticks([])
    plt.savefig('{}_segment_{}.{}'.format(args.prefix, id, args.format), format=args.format, dpi=200)
    plt.close()


def CalcDuration(df, window, id, thresholdDist=1, thresholdActive=0.95):
    ind = 0
    ext = 0
    dfs = []
    dfPosition = df.reset_index(drop=True)
    while ind + window + ext < dfPosition.index.max():
        temp = dfPosition[ind:ind + window + ext].reset_index(drop=True)
        activeRatio = len([i for i in temp.index if temp.loc[i, 'distance'] > thresholdDist]) / len(temp)
        if (temp.distance.mean() > thresholdDist) & (activeRatio > thresholdActive):
            ext += 1
        elif ext > 1:
            dfLine = pd.DataFrame({'start':ind+1, 'end':ind + window + ext, 'duration':window + ext}, index=[ind+1])
            dfs.append(dfLine)
            ind = ind + window
            ext = 0
        else:
            ind += 1
    if len(dfs) > 0:
        oDf = pd.concat(dfs, sort=False, ignore_index=True)
        return oDf
    else:
        return dfs


def returnCalcDuration(dfDur, dfDist, id):
    plt.figure(figsize=(3,1))
    sns.distplot(dfDur.duration)
    plt.savefig('{}_duration_distplot_{}.{}'.format(args.prefix, id, args.format), format=args.format, dpi=200)
    plt.close()
    listDur = dfDist.Slice.values
    activeTimePoint = []
    for i in dfDur.index:
        activeTimePoint += list(np.linspace(dfDur.loc[i, 'start'], dfDur.loc[i, 'end'], dfDur.loc[i,'duration']))
    plotDur = [i/i if i in activeTimePoint else i*0 for i in listDur]
    plt.figure(figsize=(3,2))
    _, axs = plt.subplots(nrows=2, ncols=1, figsize=(args.segment+1, 1))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axs[0].scatter('Slice', 'distance', data=dfDist, s=0.1, alpha=0.5, c="gray")
    sns.lineplot(dfDist.Slice, plotDur, drawstyle='steps-pre', color='gray', linewidth=1, ax=axs[1])
    axs[1].fill_between(dfDist.Slice, plotDur, color='gray', step='pre')
    sns.despine()
    plt.savefig('{}_locomotion_{}.{}'.format(args.prefix, id, args.format), format=args.format, dpi=200)
    plt.close()
    return dfDur.duration.median()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, metavar='file', nargs='+', help='Path to a video or a sequence of image.', default=['vtest.avi'])
    parser.add_argument('--prefix', type=str, metavar='prefix', help='Prefix of path to output files.', default='stdout')
    parser.add_argument('--algo', type=str, help='Background subtraction method.', default='KNN', choices=['MOG2', 'KNN'])
    parser.add_argument('--learningRate', metavar='float', type=float, help='Learning Rate for apply method', default=-1)
    parser.add_argument('--blurringSquare', metavar='length', type=int, help='Edge length of blurring square', default=10)
    parser.add_argument('--live', action='store_true', help='Display processing movie in live, default=False')
    parser.add_argument('--noPoint', action='store_false', help='Pointing, default=True')
    parser.add_argument('--fps', metavar='FPS', type=int, help='output FPS', default=-1)
    parser.add_argument('--analysis_range', metavar='range', nargs=2, type=int, help='a range of frames to be analyzed', default=(0, -1))
    parser.add_argument('--top', metavar='px', type=int, help='top position of frame', default=-1)
    parser.add_argument('--bottom', metavar='px', type=int, help='bottom position of frame', default=0)
    parser.add_argument('--left', metavar='px', type=int, help='left position of frame', default=0)
    parser.add_argument('--right', metavar='px', type=int, help='right position of frame', default=-1)
    parser.add_argument('--segment', metavar='int', type=int, help='how many segment used for area segment', default=-1)
    parser.add_argument('--segment_edge_length', metavar='px', type=int, help='length of segment square', default=-1)
    parser.add_argument('--window', metavar='frames', type=int, help='seed window for duration analysis', default=10)
    parser.add_argument('--format', type=str, help='output format for figures', default='png', choices=['png', 'pdf'])
    parser.add_argument('--process', type=int, help='number of process for analysis', default=1)
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
    sepDict = []
    with open('{}_dist_data.txt'.format(args.prefix), "w") as outDist:
        outDist.write('\t'.join(["id", "total_distance", "median_speed", "median_duration", "frequency"])+'\n')
        with tqdm.tqdm(total=(len(xborder)-1)*(len(yborder)-1)) as pbar:
            for i, j in itertools.product(range(len(xborder)-1), range(len(yborder)-1)):
                cut_bottom = yborder[j]
                cut_top = yborder[j+1]
                cut_left = xborder[i]
                cut_right = xborder[i+1]
                area_id = '{}_{}_{}_{}'.format(cut_left, cut_right, cut_bottom, cut_top)
                temp = df[(df.XM > cut_left) & (df.XM < cut_right) & (df.YM > cut_bottom) & (df.YM < cut_top)]
                temp.id = area_id
                sepDict.append(temp)
            for _ in Pool.imap_unordered(f, data):
            with Pool(args.process) as p:
                merged = mergePoints(temp, totalSlice)
                dfDist, totalDist, medSpeed = DistantCalc(merged, totalSlice, area_id)
                dfDur = CalcDuration(dfDist, args.window, area_id)
                if len(dfDur) > 0:
                    medDur = returnCalcDuration(dfDur, dfDist, area_id)
                else:
                    medDur = 0
                if args.segment > 0:
                    CalcAreaChange(dfDist, totalSlice, area_id, args.segment_edge_length)
                merged_list.append(dfDist)
                outDist.write('\t'.join([area_id, str(totalDist), str(medSpeed), str(medDur), str(len(dfDur))])+'\n')
                pbar.update(1)
    dfMerged = pd.concat(merged_list, axis=0, sort=False, ignore_index=True)
    dfMerged.to_csv('{}_positions.txt'.format(args.prefix), sep='\t', index=False)
    plt.scatter(dfMerged.XM, dfMerged.YM, c="gray", s=0.1)
    plt.hlines(yborder, colors="lightgray", linewidth=1, xmin=min(xborder), xmax=max(xborder))
    plt.vlines(xborder, colors="lightgray", linewidth=1, ymin=min(yborder), ymax=max(yborder))
    plt.savefig('{}_positions.{}'.format(args.prefix, args.format), format=args.format, dpi=200)
    plt.close()

    #----------------------# Pointing #----------------------#
    # Video setting
    if args.noPoint:
        print("Start pointing...")
        startSlice = 0
        for inCap in args.input:
            capture = cv.VideoCapture(cv.samples.findFileOrKeep(inCap))
            fcount = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
            if args.fps < 0:
                fps = capture.get(cv.CAP_PROP_FPS)
            else:
                fps = args.fps 
            
            fourcc = cv.VideoWriter_fourcc('m','p','4', 'v')
            oWidth = int((right-args.left) * 0.5)
            oHeight = int((top-args.bottom) * 0.5)
            outVideo = cv.VideoWriter("{}_pointed_{}".format(args.prefix, inCap.replace("avi", "mp4")), fourcc, fps, (oWidth, oHeight), isColor=False)
            with tqdm.tqdm(total=fcount) as pbar:
                while True:
                    ret, frame = capture.read()
                    if frame is None:
                        break
                    pbar.update(1)
                    frame_num = startSlice + int(capture.get(cv.CAP_PROP_POS_FRAMES))
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
                    outVideo.write(cv.resize(im_gray, dsize=(oWidth, oHeight)))

                    keyboard = cv.waitKey(30)
                    if keyboard == 'q' or keyboard == 27:
                        break
            outVideo.release()
            startSlice += fcount
