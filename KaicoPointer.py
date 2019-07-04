import os
import sys
import numpy as np
import pandas as pd
import cv2


# plot the center
def PlotCenter(frame, df):
    for _, i in df.iterrows():
            x = int(round(i["X"], None))
            y = int(round(i["Y"], None))
            frame = cv2.circle(
                frame,
                (x, y),
                5,
                color=(0, 0, 255),
                thickness=-1
                )
    return frame


# main body
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
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif form == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        print("format should be AVI or MP4")
        sys.exit()

    # read input & make output
    cap = cv2.VideoCapture(inf)
    fps = cap.get(cv2.CAP_PROP_FPS)
    outx = int(cap.get(3))
    outy = int(cap.get(4))
    out = cv2.VideoWriter(outf, fourcc, fps, (outx, outy))

    # depict the circles
    c = 0
    while(cap.isOpened()):
        ref, frame = cap.read()
        if ref:
            temp = df[df["Slice"] == c]
            c += 1
            of = PlotCenter(frame, temp)
            out.write(of)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # release videos
    cap.release
    out.release()
    cv2.destroyAllWindows()
