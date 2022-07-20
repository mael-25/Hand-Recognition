
import glob
import os
import random
import shutil
import cv2


def main(gestures=[], subsample=True, percentTrain=85, shuffle=False):
    # subsample = True

    data_dir = '/Users/mael/Dataset/Gestures/data/{}images/'.format(
        "subsample/" if subsample else "")

    for g in gestures:

        # gestures = [""]

        # shutil.rmtree()

        # os.rmdir()

        d = glob.glob(data_dir+g+"/*.jpg")
        # print(data_dir+g+"*.jpg")
        d = [x.split(data_dir+g+"/")[-1] for x in d]

        if shuffle:
            random.shuffle(d)

        t = d[:int(percentTrain*len(d)/100)]
        v = d[int(percentTrain*len(d)/100):]
        # print(d)

        os.makedirs(data_dir+"train/"+g, exist_ok=True)
        os.makedirs(data_dir+"val/"+g, exist_ok=True)

        for f in t:
            shutil.move(data_dir+g+"/"+f, data_dir+"train/"+g+"/"+f)
        for f in v:
            shutil.move(data_dir+g+"/"+f, data_dir+"val/"+g+"/"+f)

        print("{} gesture done".format(g))

        try:
            os.rmdir(data_dir+g+"/")
        except FileNotFoundError:
            pass

        # raise

        # os.makedirs(data_dir+"train", exist_ok=True)
        # os.makedirs(data_dir+"val", exist_ok=True)


if __name__ == "__main__":
    main(['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace_inverted',
         'peace', 'rock', 'stop_inverted', 'stop', 'three', 'three2', 'two_up_inverted', 'two_up'])
