from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from matplotlib import pyplot as plt
import argparse
import tqdm
import itertools


def defineBorders(data):
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


def DistantCalc(xlim, ylim, data):
    temp = data[(data.XM>xlim[0]) & (data.XM<xlim[1]) & (data.YM>ylim[0]) & (data.YM<ylim[1])]
    #print(temp.head())
    times = []
    for i in range(1, period+1):
        #print(i)
        time = temp[temp.Slice == i]
        if len(time) == 1:
            times.append(time)
        elif (len(time) == 0) & (i > 1):
            times.append(times[(i-1)-1])
        elif (len(time) == 0) & (i == 1):
            df_first = temp[temp.Slice == (i+1)]
            while(len(df_first) == 0):
                i += 1
                df_first = temp[temp.Slice == (i+1)]
            if len(df_first) > 1:
                df_first = pd.DataFrame({"Area": np.average(df_first.Area), "XM": np.average(df_first.XM), "YM": np.average(df_first.YM), "Slice": i}, index=[1])
            times.append(df_first)
        elif (len(time) > 1):
            df_multi = pd.DataFrame({"Area": np.average(time.Area), "XM": np.average(time.XM), "YM": np.average(time.YM), "Slice": i}, index=[1])
            times.append(df_multi)
    data1 = pd.concat(times, sort=False)
    #print(len(temp), len(data1), period)
    data1.Slice = range(1, period+1)
    dist = np.sqrt(\
                   np.power(data1.XM.values[0:len(data1)-1]-data1.XM.values[1:len(data1)], 2)\
                   + np.power(data1.YM.values[0:len(data1)-1]-data1.YM.values[1:len(data1)], 2)\
                  )
    data1["distance"] = [0] + list(dist)
    accdist = np.cumsum(data1.distance)
    data1["accumlated"] = accdist
    return (data1, list(accdist)[-1])


# main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot center position')
    parser.add_argument('--track', type=str, help='Path to a track data from KaicoTracker.', default='stdout.txt')
    parser.add_argument('--prefix', type=str, help='Path and prefix of output video and table.', default='stdout')
    parser.add_argument('--top', type=int, help='top position of frame', default=-1)
    parser.add_argument('--bottom', type=int, help='bottom position of frame', default=0)
    parser.add_argument('--left', type=int, help='left position of frame', default=0)
    parser.add_argument('--right', type=int, help='right position of frame', default=-1)
    args = parser.parse_args()

    data = pd.read_csv(args.track, sep="\t")
    lastSlice = data.Slice.max()
    xborder, yborder = defineBorders(data, lastSlice)
    odf_list = []
    with tqdm.tqdm((len(xborder)-1)*(len(yborder)-1)) as pbar:
        for i, j in itertools.product(range(len(xborder)-1), range(len(yborder)-1)):
            bottom = yborder[j]
            top = yborder[j+1]
            left = xborder[i]
            right = xborder[i+1]
            temp = data[(data.XM > left) & (data.XM < right) & (data.YM > bottom) & (data.YM < top)]
            odf = mergePoints(temp, lastSlice)
            #odf.to_csv('{}_positions_{}_{}_{}_{}.txt'.format(args.prefix, left, right, bottom, top), sep='\t', index=False)
            odf_list.append(odf)
            pbar.update(1)
    odata = pd.concat(odf_list, axis=0, sort=False, ignore_index=True)
    odata.to_csv('{}_positions.txt'.format(args.prefix), sep='\t', index=False)
