import os
import sys
import numpy as np
import pandas as pd
import cv2 as cv
import argparse


# plot the center
def PlotCenter(frame, df):
    for _, i in df.iterrows():
            x = int(round(i["X"], None))
            y = int(round(i["Y"], None))
            frame = cv.circle(
                frame,
                (x, y),
                5,
                color=(0, 0, 255),
                thickness=-1
                )
    return frame


# main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot center position')
    parser.add_argument('--video', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--borders', type=str, help='Path to a file designating borders', default='test.txt')
    parser.add_argument('--prefix', type=str, help='Path and prefix of output video and table.', default='stdout')
    parser.add_argument('--top', type=int, help='top position of frame', default=-1)
    parser.add_argument('--bottom', type=int, help='bottom position of frame', default=0)
    parser.add_argument('--left', type=int, help='left position of frame', default=0)
    parser.add_argument('--right', type=int, help='right position of frame', default=-1)
    parser.add_argument('--fps', type=int, help='frame rate of output video', default=-1)
    args = parser.parse_args()

    # read input & make output
    cap = cv.VideoCapture(args.video)
    if args.fps < 0:
        fps = cap.get(cv.CAP_PROP_FPS)
    else:
        fps = args.fps
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    videoW = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    videoH = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    if args.top < 0:
        top = videoH
    else:
        top = args.top
    if args.right < 0:
        right = videoW
    else:
        right = args.right
    
    out = cv.VideoWriter("{}.mp4".format(args.prefix), fourcc, fps, (right-args.left, top-args.bottom))


if __name__ == "__main__":
    # check the number of argument
    argvs = sys.argv
    if len(argvs) != 5:
        print(
            "ERROR: illegal number of arguments, \
                Usage: KaicoPointer.py input results output format"
            )
        sys.exit()
    inf = argvs[1]
    df = pd.read_csv(argvs[2])
    outf = argvs[3]
    form = argvs[4]

    # check the format
    if form == "AVI":
        fourcc = cv.VideoWriter_fourcc(*'XVID')
    elif form == "MP4":
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
    else:
        print("format should be AVI or MP4")
        sys.exit()

    # read input & make output
    cap = cv.VideoCapture(inf)
    fps = cap.get(cv.CAP_PROP_FPS)
    outx = int(cap.get(3))
    outy = int(cap.get(4))
    out = cv.VideoWriter(outf, fourcc, fps, (outx, outy))

    # depict the circles
    c = 0
    while(cap.isOpened()):
        ref, frame = cap.read()
        if ref:
            temp = df[df["Slice"] == c]
            c += 1
            of = PlotCenter(frame, temp)
            out.write(of)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # release videos
    cap.release
    out.release()
    cv.destroyAllWindows()
