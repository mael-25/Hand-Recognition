from __future__ import division, print_function
from torch.autograd import Variable

import copy
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from model import CNN

name = "mael2"

lr = 1e-3
wd = 1e-4

rate = 25
decay = 0.1
dropout = 0.1


# lr = {10:}


# import matplotlib.pyplot as plt


epochs = 50  # 25
batch_size = 16

subsample = True

data_dir = '/Users/mael/Dataset/Gestures/data/{}images'.format(
    "subsample/" if subsample else "")


plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation((-25, 25)),
        transforms.RandomResizedCrop(96, (0.85, 1), (7 / 8, 8 / 7)),
        # transforms.Resize(128),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.ToPILImage()
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomRotation((-180, 180)),
        # transforms.CenterCrop(256),
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

loaders = {
    'train': DataLoader(image_datasets["train"],
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,),

    'test': DataLoader(image_datasets['val'],
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=0),
}


def train(cnn: CNN, loss_func: nn.CrossEntropyLoss, optimizer: optim.Adam, loaders: DataLoader):
    cnn.train()

    debug = False

    for i, (images, labels) in enumerate(loaders['train']):

        b_x = Variable(images)   # batch x
        b_y = Variable(labels)   # batch y

        output = cnn(b_x)[0]
        loss: torch.Tensor = loss_func(output, b_y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        if debug:
            for y in range(len(images)):
                # print(images[y].numpy().shape)
                # print(images[y])
                a: np.ndarray = images[y].numpy()

                a *= 255
                a = a.astype(np.uint8)

                # print(a)

                a = a.transpose([1, 2, 0])
                # a = a * 255

                cv2.imshow("", a)
                cv2.waitKey(0)

    return loss

    pass


def val(cnn: CNN, loaders: DataLoader):
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        debug = False

        for x, (images, labels) in enumerate(loaders['test']):
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()

            if debug:
                for y in range(len(images)):
                    # print(images[y].numpy().shape)
                    # print(images[y])
                    a: np.ndarray = images[y].numpy()

                    a *= 255
                    a = a.astype(np.uint8)

                    # print(a)

                    a = a.transpose([1, 2, 0])
                    # a = a * 255

                    cv2.imshow("Predicted: {} Truth: {}".format(
                        pred_y[y], labels[y]), a)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            # print(type(images[0]))
            # print(pred_y, labels, pred_y == labels)
            correct += sum(pred_y == labels)
            # print(sum(pred_y == labels))
            # print(type(correct))
            total += len(labels)

        # print(total,  correct)

        accuracy = correct/total

        # print(accuracy)

        # a2 = [a for a in accuracy]
        # a3 = sum(a2)/len(a2)

        return accuracy, total


def main(epochs, rate, decay, dropout, name: str = "1"):
    cnn = CNN(dropout=dropout)

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(cnn.parameters(), lr=lr,
                           eps=1e-7, weight_decay=wd)

    hs = 0

    for e in range(epochs):
        loss = train(cnn, loss_func, optimizer, loaders)
        accuracy, total = val(cnn, loaders)

        hs = max(hs, accuracy)

        if e == 25 and hs > 50:
            return True

        print("Epoch [{:02d}/{}], Loss:{:.4f}, Accuracy: {:.5f}, Highscore: {:.5f}".format(e,
              epochs, loss.item(), accuracy, hs))

        if hs == accuracy:
            torch.save(
                cnn, '/Users/mael/Desktop/Desktop - Maël’s MacBook Air/Coding/Python/Machine learning/Gesture recognition/Logs/{}'.format(name+".pt" if name.split(".")[-1] != ".pt" else name))

            print("Saved model!")

        if (e+1) % rate == 0:
            for g in optimizer.param_groups:
                g['lr'] *= decay
            print("LR CHANGED to {}".format(optimizer.param_groups[0]['lr']))

    test(cnn, loaders)

    return False


def test(cnn: CNN, loaders=DataLoader):
    cnn.eval()
    a = time.time()
    with torch.no_grad():
        correct = 0
        total = 0
        debug = False

        for x, (images, labels) in enumerate(loaders['test']):
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()

            if debug:
                for y in range(len(images)):
                    # print(images[y].numpy().shape)
                    # print(images[y])
                    a: np.ndarray = images[y].numpy()

                    a *= 255
                    a = a.astype(np.uint8)

                    # print(a)

                    a = a.transpose([1, 2, 0])
                    # a = a * 255

                    cv2.imshow("Predicted: {} Truth: {}".format(
                        pred_y[y], labels[y]), a)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            # print(type(images[0]))
            # print(pred_y, labels, pred_y == labels)
            correct += sum(pred_y == labels)
            # print(sum(pred_y == labels))
            # print(type(correct))
            total += len(labels)

        # print(total,  correct)

        accuracy = correct/total

        # print(accuracy)

        # a2 = [a for a in accuracy]
        # a3 = sum(a2)/len(a2)

        a2 = time.time

        print("Took {} seconds".format(a2-a))

        return accuracy, total


if __name__ == "__main__":
    if os.path.exists('/Users/mael/Desktop/Desktop - Maël’s MacBook Air/Coding/Python/Machine learning/Gesture recognition/Logs/{}'.format(name+".pt" if name.split(".")[-1] != ".pt" else name)):
        f = False
        while not f:
            a = input('/Users/mael/Desktop/Desktop - Maël’s MacBook Air/Coding/Python/Machine learning/Gesture recognition/Logs/{} exists, overwrite? (y/n)'.format(
                name+".pt" if name.split(".")[-1] != ".pt" else name)).lower()
            f = True
            if a.lower().startswith('y'):
                pass
            elif a.lower().startswith("n"):
                exit()
            else:
                f = False
    a = True
    while a == True:
        a = main(epochs, rate, decay, dropout, name)
