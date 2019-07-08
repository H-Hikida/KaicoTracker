import sys
from collections import Counter
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import imagehash
from sklearn.cluster import KMeans


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
def sumContrours(contours, img):
    contours_new = []
    params = []
    for i in contours:
        # simplify contours
        epsilon = 0.01*cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        # get contours properties
        M = cv2.moments(i)
        if M["m00"] != 0:
            area = cv2.contourArea(i)
            perimeter = cv2.arcLength(i, True)
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)
            cv2.drawContours(mask, [i], 0, 255, -1)
            mean, std = cv2.meanStdDev(img, mask=mask)
            contours_new.append(approx)
            param = [
                area,
                perimeter,
                float(mean[0]),
                float(mean[1]),
                float(mean[2]),
                float(std[0]),
                float(std[1]),
                float(std[2])
            ]
            params.append(param)
            # print(mean, std)
    return contours_new, params


# clustering contours
def clusterCNT(params):
    aparams = np.array(params)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(aparams)
    labels = kmeans.labels_
    # for i in range(1,6):
    #    if labels.count(i) == larvae:
    #        key = i
    # keys = [i for i in labels if i == key]
    return len(labels), labels


# draw contours with different colors
# colors are list of integer designating the cluster
def drawCNT(img, cnt, colors):
    pallette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        ]
    c = 0
    for i in cnt:
        img = cv2.drawContours(img, i, -1, pallette[colors[c]], 2)
        c += 1
    return img


# set threshold and mask
def cutoutLarvae(hsv, larvae):
    contours = list(range(0, larvae * 11))
    mat = np.array([[0, 0, 100], [360, 100, 255]])
    # gaussian filter
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
    # collect properties of contours
    cnt_new, params = sumContrours(contours, hsv)
    # clustering based on parameter
    colnum, colors = clusterCNT(params)
    # make image with contours with different colors
    img_cont = drawCNT(res, cnt_new, colors)
    return img_cont, len(cnt_new)


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
