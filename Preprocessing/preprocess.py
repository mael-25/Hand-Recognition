import copy
import glob
import json
import os
import random
import cv2
from cv2 import ROTATE_180
import tqdm

# files = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace_inverted',
#          'peace', 'rock', 'stop_inverted', 'stop', 'three', 'three2', 'two_up_inverted', 'two_up']

files = ["fist"]

subsample = False

processAll = False

save = False

debug = False

datasetFolder = "/Users/mael/Dataset/Gestures/data{}".format(
    "/subsample" if subsample else "")

filesOLD = copy.deepcopy(files)

# class NotAFile(Exception):


if processAll:
    # [x.split(".")[0] for x in glob.glob(os.path.join("data", "origAnn", "*.json"))]
    files = glob.glob(os.path.join(datasetFolder, "origAnn", "*.json"))
    # print(files)
    raise Exception("")
    # images = [glob.glob(os.path.join("data", "images", x, "*.jpg")) for x in files]
else:
    files = filesOLD
    # images = glob.glob(os.path.join("data", "images", "*.jpg"))

for f in files:
    annotations: dict = json.load(
        open(os.path.join(datasetFolder, "origAnn", "{}.json".format(f))))
    annotationsList = list(annotations)

    os.makedirs("{}/images/{}".format(datasetFolder, f), exist_ok=True)
    os.makedirs("{}/annotations/{}".format(datasetFolder, f), exist_ok=True)

    # print(f)
    # print(annotationsList)
    t = tqdm.tqdm(range(len(annotationsList)), colour="green")
    ti = iter(t)
    for image in annotationsList:
        i = copy.deepcopy(image)
        # print("{}.jpg".format(image), end="\r")
        # i = copy.deepcopy(image)
        settings = annotations[image]
        image = os.path.join(datasetFolder, "origImg",
                             f, "{}.jpg".format(image))
        # print(image)
        img = cv2.imread(image)
        # if img == None:
        #     continue
        x1, y1 = 0, 0
        x2, y2 = 0, 0
        # print(settings)
        try:
            size = img.shape[:2]
        except AttributeError:
            next(ti)
            continue

        y1, x1 = size

        # print(settings)
        try:
            num = settings["labels"].index(f)

            x1 = int(x1*settings["bboxes"][num][0])
            y1 = int(y1*settings["bboxes"][num][1])
            x2 = int(size[1]*settings["bboxes"][0][2]+x1)
            y2 = int(size[0]*settings["bboxes"][0][3]+y1)

            w = int(size[1]*settings["bboxes"][0][2])
            h = int(size[0]*settings["bboxes"][0][3])

            if debug:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
            extx = w*0.3
            exty = h*0.3

            x1 -= extx
            y1 -= exty
            x2 += extx
            y2 += exty

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            distx = x2-x1
            disty = y2-y1

            if distx == disty:
                pass
            elif distx > disty:
                big = "x"
                d = distx-disty
                d2 = d//2
            elif distx < disty:
                big = "y"
                d = disty-distx
                d2 = d//2
            else:
                print("??? Dont know what happened ???")

            if big == "x":
                y1 -= d2
                y2 += d2

            elif big == "y":
                x1 -= d2
                x2 += d2
            dx = 0-x1
            dy = 0-y1
            dw = x2-size[1]
            dh = y2-size[0]
            cropImg = img[max(0, y1):min(size[0], y2),
                          max(0, x1):min(size[1], x2)]

            # print(max(0, dx), max(dw, 0), max(dy, 0), max(0, dh))

            cropImg = cv2.copyMakeBorder(cropImg,
                                         top=max(dy, 0),
                                         bottom=max(0, dh),
                                         left=max(0, dx),
                                         right=max(dw, 0),
                                         borderType=cv2.BORDER_REPLICATE
                                         )
            cropImg = cv2.resize(cropImg, (320, 320))
            # cv2.imshow("cropped1", cropImg)

            d = {"gesture": f, "hand": settings["leading_hand"]}

            json.dump(
                d, open("{}/annotations/{}/{}.json".format(datasetFolder, f, i), "w"))

            if debug:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255))
            # cv2.imshow("origImg", img)
            # k = cv2.waitKey(1)
            # print(k)

            cv2.imwrite("{}/images/{}/{}.jpg".format(datasetFolder,
                        f, i), cropImg, [cv2.IMWRITE_JPEG_QUALITY, 100])
            # print("images/{}/{}.jpg".format(f, i))
            # k = cv2.waitKey(0)
            # break
            # if k in [113, 81, 27]:
            #     break

        except ValueError:
            print("ERROR!!!")
            # os.remove(image)

        next(ti)

        # except Exception as E:
        #     print(E)

    # os.remove("data/annotations/{}.json".format(f))
