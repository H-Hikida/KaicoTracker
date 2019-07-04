import sys
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import imagehash


# generate hash from images
def hashSampling(frame, L, type="a"):
    img = Image.fromarray(np.uint8(frame))
    if type == "a":
        hash = imagehash.average_hash(img)
    elif type == "p":
        hash = imagehash.phash(img)
    elif type == "d":
        hash = imagehash.dhash(img)
    elif type == "w":
        hash = imagehash.whash(img)
    else:
        print("Argument error: type must be in 'a, p, d, w'")
    if hash not in L:
        L.append(hash)
        # print(hash)
        return L, True
    else:
        return L, False


# summrize contours' information
def sumContrours(contours):
    contours_new = []
    areas = []
    cxs = []
    cys = []
    for i in contours:
        epsilon = 0.01*cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        contours_new.append(approx)
        area = cv2.contourArea(i)
        areas.append(area)
        M = cv2.moments(i)
        if M["m00"] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx = None
            cy = None
        cxs.append(cx)
        cys.append(cy)
    df = pd.DataFrame({"area": areas, "X": cxs, "Y": cys})
    return contours_new, df


# virtually crop image with biggest contour
def vCropBiggest(contours, df):
    areas = list(df["area"].values)
    b_index = areas.index(max(areas))
    biggestContour = contours[b_index]
    L = []
    for i in biggestContour:
        L.append(i[0])
    cdf = pd.DataFrame.from_records(L)
    xmin, xmax = np.float64(min(cdf[0].values)), np.float64(max(cdf[0].values))
    ymin, ymax = np.float64(min(cdf[1].values)), np.float64(max(cdf[1].values))
    tags = []
    for _, i in df.iterrows():
        if (np.isnan(i["X"])) | (np.isnan(i["Y"])):
            tag = False
        else:
            if (xmin < i["X"] < xmax) & (ymin < i["Y"] < ymax):
                tag = True
            else:
                tag = False
        tags.append(tag)
    df["tag"] = tags
    new_df = df[df["tag"]]
    new_contours = list(np.array(contours)[np.array(tags)])
    return new_contours, new_df


# selection based on area
def selectLarvae(contours, df):
    largest = max(df["area"])
    tags = []
    for i in df["area"]:
        tag = largest/1000 < i < largest/50
        tags.append(tag)
    new_contours = list(np.array(contours)[np.array(tags)])
    return new_contours


# set threshold and mask
def cutoutLarvae(hsv, larvae):
    contours = list(range(0, larvae * 11))
    mat = np.array([[0, 0, 100], [360, 100, 255]])
    NowTraining = True
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    # make masked data
    lower = np.array(mat[0], dtype=np.uint8)
    upper = np.array(mat[1], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # make bitwise image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # determine contours
    contours, hierarchy = cv2.findContours(
                    mask,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                    )
    cnew, df = sumContrours(contours)
    contours_temp, new_df = vCropBiggest(cnew, df)
    contours_larvae = selectLarvae(contours_temp, new_df)
    # make image with contours
    img_cont = cv2.drawContours(res, contours_larvae, -1, (0, 0, 255), 1)
    return img_cont, len(contours_larvae)


# main body
if __name__ == "__main__":
    # check the number of argument
    argvs = sys.argv
    if len(argvs) != 3:
        print(
            "ERROR: illegal number of arguments, \
                Usage: KaicoSeeker.py input_path output_path"
            )
        sys.exit()
    in_path = argvs[1]
    o_path = argvs[2]

    # read the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(in_path)
    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(length)

    # generate output
    outx, outy = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(o_path, fourcc, 20.0, (outx, outy))

    # sampling based on hash
    hash_L = []
    while(cap.isOpened()):
        ref, frame = cap.read()
        if ref:
            hash_L, res = hashSampling(frame, hash_L)
            if res:
                # convert to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                img_cont, length = cutoutLarvae(hsv, 9)
                print(length)
                out.write(img_cont)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

cap.release
out.release()
cv2.destroyAllWindows()
