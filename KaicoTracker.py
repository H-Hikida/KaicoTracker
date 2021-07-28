from __future__ import print_function
import sys
import re
import warnings
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.collections import PatchCollection
import seaborn as sns
import cv2 as cv
import itertools


def printCounter(c, total, progress):
    if c*100//total > progress:
        print('{}% Done'.format(str(progress)))
        progress += 10
    c += 1
    return c, progress


def startCapture(inCap):
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(inCap))
    fcount = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened():
        print('Unable to open: ' + inCap)
        exit(0)
    return capture, fcount


def borderSelection(df, ncols, nrows):
    cropBorders = {}
    for i in ['XM', 'YM']:
        if i == 'XM':
            iniVal = int(df['XM'].min())
            endVal = int(df['XM'].max())
            div = ncols
        else:
            iniVal = int(df['YM'].min())
            endVal = int(df['YM'].max())
            div = nrows
        if div < 0:
            cropBorders[i] = (int(iniVal), int(endVal))
            continue
        Slices = np.linspace(iniVal, endVal, num=endVal-iniVal)
        kernel = stats.gaussian_kde(df[i])
        estimates = kernel(Slices)
        mins = [iniVal] + [int(i) for i in signal.argrelmin(estimates)[0].tolist() if (i > iniVal) & (i < endVal)] + [endVal]
        if len(mins) > div + 1:
            volume = []
            for j in range(len(mins)-1):
                volume.append(len(df[(df[i] > mins[j]) & (df[i] < mins[j+1])]))
            for j in range(1, len(volume)-1):
                if (volume[j] < volume[j-1] * args.cropThreshold) & (volume[j] < volume[j+1] * args.cropThreshold):
                    volume[j] = -1
            volume += [1]
            borders = [mins[j] for j in range(len(mins)) if volume[j] > 0]
            startsList = borders[:-div]
            endsList = borders[div:]
            frameMatrix = pd.DataFrame({'start': startsList, 'end': endsList, 'count': [len(df[(df[i]>startsList[j]) & (df[i]<endsList[j])]) for j in range(0, len(borders)-div)]})
            cropBorders[i] = (frameMatrix.loc[frameMatrix['count'].idxmax(), 'start'], frameMatrix.loc[frameMatrix['count'].idxmax(), 'end'])
        else:
            cropBorders[i] = (int(iniVal), int(endVal))
    return cropBorders['XM'], cropBorders['YM']


def autoCrop(capture):
    dfs =[]
    processFrames = min(args.autoCropPreAnalysis, fcount)
    print('Tracking {} frames'.format(str(processFrames)))
    c, progress = 0, 10
    while True:
        _, frame = capture.read()
        if (frame is None) | (capture.get(cv.CAP_PROP_POS_FRAMES) > processFrames):
            break
        frame_num = cv.CAP_PROP_POS_FRAMES
        c, progress = printCounter(c, processFrames, progress)
        _, contours, _ = subBack(frame)
        for i in contours:
            # Minimum Enclosing Circle
            (x,y),radius = cv.minEnclosingCircle(i)
            # make DataFrame row
            d = {'Slice': [int(frame_num)], 'XM': [int(x)], "YM": [int(y)], 'radius': [int(radius)]}
            dfs.append(pd.DataFrame(data=d, index=[0]))
    print('100% Done, Completed!')
    print('Determing borders..')
    df = pd.concat(dfs, ignore_index=True, sort=False)
    xlims, ylims = borderSelection(df, ncols=args.autoCrop[0], nrows=args.autoCrop[1])
    return xlims, ylims


def setCropArea(capture):
    if args.autoCrop != (-1, -1):
        top = ylims[1]
    elif args.top < 0:
        top = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    else:
        top = args.top
    if args.autoCrop != (-1, -1):
        right = xlims[1]
    elif args.right < 0:
        right = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    else:
        right = args.right
    return top, right


def subBack(frame):
    fgMask = backSub.apply(frame, learningRate=args.learningRate)
    fgMaskBlur = cv.blur(fgMask, (args.blurringSquare, args.blurringSquare))
    canny_output = cv.Canny(fgMaskBlur, 100, 100 * 2)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return fgMaskBlur, contours, hierarchy


def defineKDE(df, columnName):
    smin = int(df[columnName].min())
    smax = int(df[columnName].max())
    Slices = np.linspace(smin, smax, num=smax-smin+1)
    kernel = stats.gaussian_kde(df[columnName])
    estimates = kernel(Slices)
    return [smin] + [i + smin for i in signal.argrelmin(estimates)[0].tolist()] + [smax], Slices, estimates


def defineBorders(df):
    borders = {}
    for i in ['XM', 'YM']:
        outL, Slices, estimates = defineKDE(df, i)
        borders[i] = outL
        plt.scatter(x=Slices, y=estimates, s=0.1, c="gray")
        ymin, ymax = plt.ylim()
        plt.vlines(borders[i], ymin=ymin, ymax=ymax, color='lightgray', zorder=1)
        plt.xticks(borders[i],borders[i])
        plt.ylim(ymin, ymax)
        sns.despine()
        plotTight(args.format)
        plt.savefig('{}_{}_kde.{}'.format(args.prefix, i, args.format), format=args.format, dpi=200)
        plt.close('all')
    return borders['XM'], borders['YM']


def adjustBorders(df, xborder, yborder):
    newXborder = []
    newYborder = []
    for i in range(len(xborder)-1):
        temp = df[(df['XM'] > xborder[i]) & (df['XM'] < xborder[i+1])]
        if len(temp) == 0:
            newYborder.append(yborder)
            continue
        outL, _, _ = defineKDE(temp, 'YM')
        if len(outL) == len(yborder):
            newYborder.append(outL)
        else:
            newYborder.append(yborder)
    for i in range(len(yborder)-1):
        temp = df[(df['YM'] > yborder[i]) & (df['YM'] < yborder[i+1])]
        if len(temp) == 0:
            newXborder.append(xborder)
            continue
        outL, _, _ = defineKDE(temp, 'XM')
        if len(outL) == len(xborder):
            newXborder.append(outL)
        else:
            newXborder.append(xborder)
    return newXborder, newYborder


def mergePoints(df):
    outDfs = []
    for i in analysisRange:
        points = df[df.Slice == i]
        if len(points) > 0:
            outX = int(points.XM.mean())
            outY = int(points.YM.mean())
            outDfs.append(pd.DataFrame({'Slice': i, 'XM': outX, 'YM': outY, 'file': points.file.tolist()[0], 'id': df['id'].iloc[0]}, index=[i]))
    if len(outDfs) > 0:
        return pd.concat(outDfs, sort=False)


def DistantCalc(data):
    times = []
    for i in analysisRange:
        time = data[data.Slice == i]
        if len(time) == 1:
            times.append(time)
        elif (len(time) == 0) & (i > initialAnalysis):
            times.append(times[(i-1)-initialAnalysis])
        elif (len(time) == 0) & (i == initialAnalysis):
            df_first = data[data.Slice == (i+1)]
            while(len(df_first) == 0):
                i += 1
                df_first = data[data.Slice == (i+1)]
            times.append(df_first)
    data1 = pd.concat(times, sort=False)
    data1.loc[:,'id'] = data['id'].iloc[0]
    data1.Slice = analysisRange
    data1.index = analysisRange
    dist = np.sqrt(\
                   np.power(data1.XM.values[0:len(data1)-1]-data1.XM.values[1:len(data1)], 2)\
                   + np.power(data1.YM.values[0:len(data1)-1]-data1.YM.values[1:len(data1)], 2)\
                  )
    data1["distance"] = [0] + list(dist)
    accdist = np.cumsum(data1.distance)
    data1["accumlated"] = accdist
    plt.figure(figsize=(6,2))
    plt.scatter(x='Slice', y='distance', data=data1, s=0.1, alpha=0.5, c="gray")
    plt.ylim(-1,20)
    sns.despine()
    plotTight(args.format)
    plt.savefig('{}_distance_{}.{}'.format(args.prefix, data1['id'].iloc[0], args.format), format=args.format, dpi=200)
    plt.close('all')
    plt.figure(figsize=(6,2))
    sns.lineplot(x='Slice', y='accumlated', data=data1, linewidth=1, color='gray')
    sns.despine()
    plotTight(args.format)
    plt.savefig('{}_accdist_{}.{}'.format(args.prefix, data1['id'].iloc[0], args.format), format=args.format, dpi=200)
    plt.close('all')
    active = data1[data1.distance>0]
    if len(active) > 1:
        sns.displot(data=active, x='distance', color= 'gray', height=2, aspect=3, kde=True)
        plotTight(args.format)
        plt.savefig('{}_speed_distplot_{}.{}'.format(args.prefix, data1['id'].iloc[0], args.format), format=args.format, dpi=200)
        plt.close('all')
    return (data1, list(accdist)[-1], active.distance.median())


def CalcAreaChange(df, length=-1):
    numSegment = (endAnalysis - initialAnalysis) // args.segment + 1
    _, axs = plt.subplots(nrows=1, ncols=numSegment, figsize=(numSegment, 1))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    Xcen = df.XM.mean()
    Ycen = df.YM.mean()
    if length < 0:
        Xstd = df.XM.std()
        Ystd = df.YM.std()
        length = max(Xstd, Ystd, 1) * 3
    for k in range(0, numSegment):
        part = df[(df.Slice > args.segment * k) & (df.Slice <= args.segment * (k+1))]
        axs[k].set_xlim(Xcen-length, Xcen+length)
        axs[k].set_ylim(Ycen-length, Ycen+length)
        axs[k].scatter(part.XM, part.YM, s=0.1, alpha=0.5, c='gray')
        axs[k].set_xticklabels([])
        axs[k].set_yticklabels([])
        axs[k].set_xticks([])
        axs[k].set_yticks([])
    plt.savefig('{}_segment_{}.{}'.format(args.prefix, df['id'].iloc[0], args.format), format=args.format, dpi=200)
    plt.close('all')


def CalcDuration(inf, window, thresholdDist=1, thresholdActive=0.95):
    dfs = []
    aid = inf['id'].iloc[0]
    dfPosition = inf.set_index('Slice', drop=True)
    #print(dfPosition.index.min(), dfPosition.index.max(), '\n', dfPosition.head())
    ind = dfPosition.index.min()
    ext = 0
    #print(dfPosition[28790:28810])
    while ind + window + ext < dfPosition.index.max():
        temp = dfPosition.loc[ind:ind + window + ext]
        #if len(temp) == 0:
        #    print(ind, window, ext)
        activeRatio = len([i for i in temp.index if temp.loc[i, 'distance'] > thresholdDist]) / len(temp)
        if (temp.distance.mean() > thresholdDist) & (activeRatio > thresholdActive):
            ext += 1
        elif ext > 1:
            dfLine = pd.DataFrame({'start':ind+1, 'end':ind + window + ext, 'duration':window + ext, 'id': aid}, index=[ind+1])
            dfs.append(dfLine)
            ind = ind + window + ext
            ext = 0
        else:
            ind += 1
    if ext > 1:
        dfLine = pd.DataFrame({'start':ind+1, 'end':ind + window + ext, 'duration':window + ext, 'id': aid}, index=[ind+1])
        dfs.append(dfLine)
    if len(dfs) > 0:
        oDf = pd.concat(dfs, sort=False, ignore_index=True)
        return oDf
    else:
        emptyDf = pd.DataFrame({'start':0, 'end':0, 'duration':0, 'id': aid}, index=[ind+1])
        return emptyDf


def plotDuration(dfDur, dfDist):
    sns.displot(data=dfDur, x='duration', color= 'gray', height=2, aspect=3, kde=True)
    plt.savefig('{}_duration_distplot_{}.{}'.format(args.prefix, dfDist['id'].iloc[0], args.format), format=args.format, dpi=200)
    plt.close('all')
    listDur = dfDist.Slice.values
    activeTimePoint = []
    for i in dfDur.index:
        activeTimePoint += list(np.arange(dfDur.loc[i, 'start'], dfDur.loc[i, 'end']))
    plotDur = [1 if i in activeTimePoint else 0 for i in listDur]
    _, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 1), gridspec_kw={'height_ratios':[4,1]})
    if args.format == 'png':
        plt.subplots_adjust(wspace=0, hspace=0.1, bottom=0.25, left=0.03, right=0.99)
    else:
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0.1)
    axs[0].scatter(x=dfDist.Slice, y=dfDist.distance, data=dfDist, s=0.1, alpha=0.5, c="gray")
    axs[0].set_ylim(0,20)
    axs[0].set_xticks([])
    axs[0].set_xlim(listDur.min(), listDur.max())
    sns.lineplot(x=dfDist.Slice, y=plotDur, drawstyle='steps-pre', color='gray', linewidth=1, ax=axs[1])
    axs[1].fill_between(x=dfDist.Slice, y1=plotDur, color='gray', step='pre')
    axs[1].set_ylim(0,1)
    axs[1].set_yticks([])
    axs[1].set_xlabel("")
    axs[1].set_xticks(range(0, listDur.max()-listDur.min(), args.segment))
    axs[1].set_xticklabels([i*args.lapse for i in range(0, listDur.max()-listDur.min(), args.segment)])
    axs[1].set_xlim(listDur.min(), listDur.max())
    sns.despine()
    plt.savefig('{}_locomotion_{}.{}'.format(args.prefix, dfDist['id'].iloc[0], args.format), format=args.format, dpi=200)
    plt.close('all')
    return dfDur.duration.median()


def AnalyzeData(df): 
    merged = mergePoints(df)
    dfDist, totalDist, medSpeed = DistantCalc(merged)
    dfDur = CalcDuration(dfDist, args.window)
    if len(dfDur) > 1:
        medDur = plotDuration(dfDur, dfDist)
    else:
        medDur = 0
    if args.segment > 0:
        CalcAreaChange(dfDist, args.segment_edge_length)
    return '\t'.join([df['id'].iloc[0], str(totalDist), str(medSpeed), str(medDur), str(len(dfDur))])+'\n', dfDist, dfDur


def plotTight(oFormat):
    if oFormat == 'png':
        plt.tight_layout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program consitutes of three steps, \
                                                    tracking learval movements from video, \
                                                    analyzing tracking data and calculate each locomotory parameter, \
                                                    pointing the larval position onto the original video (confirmation of result)')
    parser.add_argument('--input', type=str, metavar='file', nargs='+', help='Path to a video or a sequence of image.', default=['vtest.avi'])
    parser.add_argument('--prefix', type=str, metavar='prefix', help='Prefix of path to output files.', default='stdout')
    parser.add_argument('--algo', type=str, help='Background subtraction method.', default='KNN', choices=['MOG2', 'KNN'])
    parser.add_argument('--learningRate', metavar='float', type=float, help='Learning Rate for applied method', default=-1)
    parser.add_argument('--blurringSquare', metavar='length', type=int, help='Edge length of blurring square', default=10)
    parser.add_argument('--lapse', metavar='seconds', type=int, help='Time-lapse interval', default=3)
    parser.add_argument('--live', action='store_true', help='Display processing movie, default=False')
    parser.add_argument('--noPoint', action='store_false', help='Pointing stage will be skipped, default=True')
    parser.add_argument('--onlyTracking', action='store_true', help='Only tracking stage will be done, default=False')
    parser.add_argument('--skipTracking', action='store_true', help='Skip tracking stage, default=False \
                                                                    --trackingResult should be specified.')
    parser.add_argument('--complexBorder', action='store_true', help='If True, complex border determination was turned on, default=False')
    parser.add_argument('--trackingResult', type=str, metavar='file', help='Path to output files of Tracking. If not specified, {prefix}.txt', default=None)
    parser.add_argument('--fps', metavar='FPS', type=int, help='output FPS', default=-1)
    parser.add_argument('--analysis_range', metavar=('start', 'end'), nargs=2, type=int, help='a range of frames to be analyzed', default=(1, -1))
    parser.add_argument('--autoCrop', type=int, metavar=('columns', 'rows'), nargs=2, help='set coloumns and rows for cropping, set -1 for non-specified', default=(-1, -1))
    parser.add_argument('--autoCropPreAnalysis', metavar='Preanalyezed_frames', type=int, help='the number of preanalyzed frames', default=1000)
    parser.add_argument('--top', metavar='px', type=int, help='top position of frame', default=-1)
    parser.add_argument('--bottom', metavar='px', type=int, help='bottom position of frame', default=0)
    parser.add_argument('--left', metavar='px', type=int, help='left position of frame', default=0)
    parser.add_argument('--right', metavar='px', type=int, help='right position of frame', default=-1)
    parser.add_argument('--segment', metavar='int', type=int, help='the length of segment used for area segmentation', default=-1)
    parser.add_argument('--segment_edge_length', metavar='px', type=int, help='length of segment square', default=-1)
    parser.add_argument('--window', metavar='frames', type=int, help='seed window for duration analysis', default=10)
    parser.add_argument('--cropThreshold', metavar='ratio', type=float, help='Threshold to cut inappropreate borders', default=0.01)
    parser.add_argument('--format', type=str, help='output format for figures', default='png', choices=['png', 'pdf'])
    args = parser.parse_args()


    #----------------------# Tracking #----------------------#
    if args.skipTracking:
        print('Tracking stage was skipped')
    else:
        if args.algo == 'KNN':
            backSub = cv.createBackgroundSubtractorKNN()
        else:
            backSub = cv.createBackgroundSubtractorMOG2()
        backSub.setDetectShadows(False)
        # Pre-analysis for autocrop
        if args.autoCrop != (-1, -1):
            print('Pre-analysis for autocrop')
            totalSlice = 0
            capture, fcount = startCapture(args.input[0])
            xlims, ylims = autoCrop(capture)
            print('Cropped into l{}, r{}, b{}, t{}'.format(str(xlims[0]), str(xlims[1]), str(ylims[0]), str(ylims[1])))
        
        print('Start Tracking')

        df_captures = []
        startSlice = 0
        for inCap in args.input:
            print("Video {}".format(str(args.input.index(inCap)+1))) 
            capture, fcount = startCapture(inCap)
            top, right = setCropArea(capture)
            dfs =[]
            c, progress = 0, 10
            while True:
                ret, frame = capture.read()
                if frame is None:
                    break
                frame_num = int(capture.get(cv.CAP_PROP_POS_FRAMES)) + startSlice
                c, progress = printCounter(c, fcount, progress)

                # cropping & background subtraction & blurring foreground & contour detection
                if args.autoCrop == (-1, -1):
                    frameCropped = frame[args.bottom: top, args.left: right]
                else:
                    frameCropped = frame[ylims[0]:ylims[1], xlims[0]:xlims[1]]
                fgMaskBlur, contours, hierarchy = subBack(frameCropped)

                # display current frame #
                cv.rectangle(frameCropped, (10, 2), (100,20), (255,255,255), -1)
                cv.putText(frameCropped, str(frame_num), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                
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

            print('100% Done, Completed!')
            df_temp = pd.concat(dfs, sort=False)
            df_temp["file"] = inCap
            df_captures.append(df_temp)
            startSlice += fcount
        cv.destroyAllWindows()
        df = pd.concat(df_captures, sort=False, ignore_index=True)
        df.to_csv('{}.txt'.format(args.prefix), sep='\t', index=False)
        totalSlice = startSlice
        # if onlyTracking is specified, the program will be quit.
        if args.onlyTracking:
            sys.exit()

    #----------------------# Analyzing #----------------------#
    if args.skipTracking:
        if args.trackingResult == None:
            df = pd.read_csv('{}.txt'.format(args.prefix), sep='\t')
        else:
            df = pd.read_csv(args.trackingResult, sep='\t')
    warnings.simplefilter('ignore', category='FutureWarning')
    print("Start analyzing...")
    # Distant calculation
    sepList = []
    initialAnalysis = args.analysis_range[0]
    if args.analysis_range[1] > 0:
        endAnalysis = args.analysis_range[1]
    else:
        endAnalysis = df.Slice.max()
    dfAnalysis = df[(df.Slice > initialAnalysis) & (df.Slice < endAnalysis)]
    analysisRange = range(initialAnalysis, endAnalysis)
    # border determination
    xlims, ylims = borderSelection(dfAnalysis, args.autoCrop[0], args.autoCrop[1])
    print("Analysis is confined to l{}, r{}, b{}, t{}".format(xlims[0], xlims[1], ylims[0], ylims[1]))
    dfAnalysis = dfAnalysis[(dfAnalysis['XM'] > xlims[0]) & (dfAnalysis['XM'] < xlims[1]) & (dfAnalysis['YM'] > ylims[0]) & (dfAnalysis['YM'] < ylims[1])]
    xborder, yborder = defineBorders(dfAnalysis)
    if args.complexBorder:
        newXborder, newYborder = adjustBorders(dfAnalysis, xborder, yborder)
    for i, j in itertools.product(range(len(xborder)-1), range(len(yborder)-1)):
        if not args.complexBorder:
            cBottom = yborder[j]
            cTop = yborder[j+1]
            cLeft = xborder[i]
            cRight = xborder[i+1]
        else:
            cBottom = newYborder[i][j]
            cTop = newYborder[i][j+1]
            cLeft = newXborder[j][i]
            cRight = newXborder[j][i+1]
        area_id = '{}_{}_{}_{}'.format(cLeft, cRight, cBottom, cTop)
        temp = dfAnalysis[(dfAnalysis.XM > cLeft) & (dfAnalysis.XM < cRight) & (dfAnalysis.YM > cBottom) & (dfAnalysis.YM < cTop)].copy()
        temp['id'] = area_id
        sepList.append(temp)
    print("Start Calculating...")
    oLines = []
    merged_list = []
    Durations = []
    c, progress = 0, 10
    for i in sepList:
        c, progress = printCounter(c, (len(xborder)-1)*(len(yborder)-1), progress)
        if len(i) == 0:
            continue
        oLine, oDfDist, dfDur = AnalyzeData(i)
        oLines.append(oLine)
        merged_list.append(oDfDist)
        Durations.append(dfDur)
    print('100% Done, Completed!')
    with open('{}_dist_data.txt'.format(args.prefix), "w") as outDist:
        outDist.write('\t'.join(["id", "total_distance", "median_speed", "median_duration", "frequency"])+'\n')
        for i in oLines:
            outDist.write(i)
    dfMerged = pd.concat(merged_list, axis=0, sort=False, ignore_index=True)
    dfMerged.to_csv('{}_positions.txt'.format(args.prefix), sep='\t', index=False)
    durMerged = pd.concat(Durations, axis=0, sort=False, ignore_index=True)
    durMerged.to_csv('{}_durations.txt'.format(args.prefix), sep='\t', index=False)
    _, ax = plt.subplots()
    ax.scatter(dfMerged.XM, dfMerged.YM, c="gray", s=0.1)
    if not args.complexBorder:
        plt.hlines(yborder, colors="lightgray", linewidth=1, xmin=min(xborder), xmax=max(xborder))
        ax.set_xticks(xborder)
        ax.set_xticklabels(xborder)
        plt.vlines(xborder, colors="lightgray", linewidth=1, ymin=min(yborder), ymax=max(yborder))
        ax.set_yticks(yborder)
        ax.set_yticklabels(yborder)
    else:
        rects = []
        for i, j in itertools.product(range(len(xborder)-1), range(len(yborder)-1)):
            cBottom = newYborder[i][j]
            cHeight = newYborder[i][j+1] - newYborder[i][j]
            cLeft = newXborder[j][i]
            cWidth = newXborder[j][i+1] - newXborder[j][i]
            rect = mpatches.Rectangle((cLeft, cBottom), width=cWidth, height=cHeight, fill=True)
            rects.append(rect)
        collection = PatchCollection(rects, alpha=0.2, color='lightgray')
        ax.add_collection(collection)
    plt.savefig('{}_positions.{}'.format(args.prefix, args.format), format=args.format, dpi=200)
    plt.close('all')

    #----------------------# Pointing #----------------------#
    # Video setting
    if args.noPoint:
        print("Start pointing...")
        startSlice = 0
        for inCap in args.input:
            capture, fcount = startCapture(inCap)
            print("Video {}".format(str(args.input.index(inCap)+1))) 
            if args.fps < 0:
                fps = capture.get(cv.CAP_PROP_FPS)
            else:
                fps = args.fps
            top, right = setCropArea(capture)
            fourcc = cv.VideoWriter_fourcc('m','p','4', 'v')
            oWidth = int((right-args.left) * 0.5)
            oHeight = int((top-args.bottom) * 0.5)
            outVideo = cv.VideoWriter("{}_pointed_{}.mp4".format(args.prefix, str(args.input.index(inCap)+1)), fourcc, fps, (oWidth, oHeight), isColor=False)
            c, progress = 0, 10
            while True:
                ret, frame = capture.read()
                if frame is None:
                    break
                c, progress = printCounter(c, fcount, progress)
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
            print('100% Done, Completed!')
        cv.destroyAllWindows()
