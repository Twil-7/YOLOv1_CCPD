import os
import cv2
import numpy as np
import random

def read_path_and_coordinate():
    data_x = []
    data_y = []
    filename = os.listdir('crop_ccpd')
    filename.sort()
    for name in filename:

        path = 'crop_ccpd/' + name

        obj1 = name.split('.')
        obj2 = obj1[0].split('_')

        x1 = int(obj2[1])
        y1 = int(obj2[2])
        x2 = int(obj2[3])
        y2 = int(obj2[4])

        # img = cv2.imread(path)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.namedWindow("Image1")
        # cv2.imshow("Image1", img)
        # cv2.waitKey(0)

        data_x.append(path)
        data_y.append([x1, y1, x2, y2])

    print('ImagePath and Coordinate have been downloaded ! ')
    return data_x, data_y


def make_data():

    data_x, data_y = read_path_and_coordinate()

    np.random.seed(1)
    index = np.arange(0, len(data_x))
    np.random.shuffle(index)
    data_x = [data_x[k] for k in index]
    data_y = [data_y[k] for k in index]

    n = len(data_x)
    train_x = data_x[0:3000]
    train_y = data_y[0:3000]
    test_x = data_x[3000:n]
    test_y = data_y[3000:n]

    return train_x, train_y, test_x, test_y


make_data()