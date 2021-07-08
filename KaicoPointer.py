from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from matplotlib import pyplot as plt
import cv2 as cv
import argparse
import tqdm
import itertools


def defineBorders(data, lastSlice):
    borders = {}
    for i in ['XM', 'YM']:
        Slices = np.linspace(1, int(data[i].max()), num=int(data[i].max()))
        kernel = stats.gaussian_kde(data[i])
        estimates = kernel(Slices)
        borders[i] = [0] + signal.argrelmin(estimates)[0].tolist() + [data[i].max()]
        print(i, borders[i])
        plt.scatter(Slices, estimates, s=0.1, c="gray")
        plt.savefig('{}_{}_kde.png'.format(args.prefix, i), format="png", dpi=200)
        plt.close()
    return borders['XM'], borders['YM']


def mergePoints(df, lastSlice):
    outDfs = []
    for i in range(1,lastSlice):
        points = df[df.Slice == i]
        if len(points) > 0:
            outX = int(points.XM.mean())
            outY = int(points.YM.mean())
            outDfs.append(pd.DataFrame({'Slice': i, 'XM': outX, 'YM': outY}, index=[i]))
    return pd.concat(outDfs, sort=False)


# main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot center position')
    parser.add_argument('--video', type=str, help='Path to a video or a sequence of image.', default='NA')
    parser.add_argument('--track', type=str, help='Path to a track data from KaicoTracker.', default='stdout.txt')
    parser.add_argument('--prefix', type=str, help='Path and prefix of output video and table.', default='stdout')
    parser.add_argument('--live', action='store_true', help='Display processing movie in live, default=False')
    parser.add_argument('--top', type=int, help='top position of frame', default=-1)
    parser.add_argument('--bottom', type=int, help='bottom position of frame', default=0)
    parser.add_argument('--left', type=int, help='left position of frame', default=0)
    parser.add_argument('--right', type=int, help='right position of frame', default=-1)
    parser.add_argument('--fps', type=int, help='frame rate of output video', default=-1)
    args = parser.parse_args()

    data = pd.read_csv(args.track, sep="\t")
    lastSlice = data.Slice.max()
    xborder, yborder = defineBorders(data, lastSlice)
    odf_list = []
    for i, j in itertools.product(range(len(xborder)-1), range(len(yborder)-1)):
        bottom = yborder[j]
        top = yborder[j+1]
        left = xborder[i]
        right = xborder[i+1]
        temp = data[(data.XM > left) & (data.XM < right) & (data.YM > bottom) & (data.YM < top)]
        odf = mergePoints(temp, lastSlice)
        odf.to_csv('{}_positions_{}_{}_{}_{}.txt'.format(args.prefix, left, right, bottom, top), sep='\t', index=False)
        odf_list.append(odf)
    odata = pd.concat(odf_list, axis=0, sort=False, ignore_index=True)
    odata.to_csv('{}_positions.txt'.format(args.prefix), sep='\t', index=False)

    if args.video != 'NA':
        # read input & make output
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.video))
        fcount = capture.get(cv.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened():
            print('Unable to open: ' + args.video)
            exit(0)

        if args.fps < 0:
            fps = capture.get(cv.CAP_PROP_FPS)
        else:
            fps = args.fps

        # set Movies size
        videoW = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        videoH = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        if args.top < 0:
            top = videoH
        else:
            top = args.top
        if args.right < 0:
            right = videoW
        else:
            right = args.right
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter("{}.mp4".format(args.prefix), fourcc, fps, (right-args.left, top-args.bottom))

        with tqdm.tqdm(total=fcount) as pbar:
            while True:
                ret, frame = capture.read()
                if frame is None:
                    break
                pbar.update(1)
                frame_num = capture.get(cv.CAP_PROP_POS_FRAMES)
                frameCropped = frame[args.bottom: top, args.left: right]
                temp = odata[odata.Slice == frame_num]
                if len(temp) > 0:
                    for i in temp.index:
                        x = int(temp.loc[i, 'XM'])
                        y = int(temp.loc[i, 'YM'])
                        frameCropped = cv.circle(frameCropped, (x, y), 5, color=(0, 0, 255), thickness=-1)
                    if args.live:
                        cv.imshow('out', frameCropped)
                    out.write(frameCropped)

                keyboard = cv.waitKey(30)
                if keyboard == 'q' or keyboard == 27:
                    break
