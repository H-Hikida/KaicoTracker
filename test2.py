from __future__ import print_function
import sys
import warnings
import random
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from matplotlib import pyplot as plt
import seaborn as sns
import cv2 as cv
import tqdm
import itertools

def startCapture(inCap):
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(inCap))
    fcount = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened():
        print('Unable to open: ' + inCap)
        exit(0)
    return capture, fcount

def subBack(frame):
    fgMask = backSub.apply(frame, learningRate=0.8)
    fgMaskBlur = cv.blur(fgMask, (10, 10))
    canny_output = cv.Canny(fgMaskBlur, 100, 100 * 2)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return fgMaskBlur, contours, hierarchy

backSub = cv.createBackgroundSubtractorKNN()
backSub.setDetectShadows(False)
print('Pre-analysis for autocrop')
totalSlice = 0
capture, fcount = startCapture('/Users/hikidahiroyuki/info/CLS/KaicoSeeker/vtest_color_200.avi')
dfs =[]
while True:
    ret, frame = capture.read()
    frame_num = cv.CAP_PROP_POS_FRAMES
    if (frame is None) | (capture.get(cv.CAP_PROP_POS_FRAMES) > min(1000, fcount)):
        break
    fgMaskBlur, contours, hierarchy = subBack(frame)
    for i in contours:
        # Minimum Enclosing Circle
        (x,y),radius = cv.minEnclosingCircle(i)
        center = (int(x),int(y))
        radius = int(radius)
        # make DataFrame row
        d = {'Slice': [int(frame_num)], 'XM': [int(x)], "YM": [int(y)], 'radius': [int(radius)]}
        dfs.append(pd.DataFrame(data=d, index=[0]))
df = pd.concat(dfs, ignore_index=True, sort=False)
borders = {}
for i in ['XM', 'YM']:
    Slices = np.linspace(1, int(df[i].max()), num=int(df[i].max()))
    kernel = stats.gaussian_kde(df[i])
    estimates = kernel(Slices)
    mins = signal.argrelmin(estimates)[0].tolist()
    maxs = signal.argrelmax(estimates)[0].tolist()
    minDf = pd.DataFrame([(mins[i], estimates[(mins[i])], 'argmin') for i in range(len(mins))])
    maxDf = pd.DataFrame([(maxs[i], estimates[(maxs[i])], 'argmax') for i in range(len(maxs))])
    print(pd.concat([minDf, maxDf], sort=False, ignore_index=True))
