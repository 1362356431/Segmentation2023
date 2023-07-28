import glob
import os

import cv2
import numpy as np


def npz(im, la, s):
    images_path = im
    labels_path = la
    path2 = s
    images = os.listdir(images_path)
    for s in images:
        image_path = os.path.join(images_path, s)
        print(image_path)
        label_path = os.path.join(labels_path, s)
        label_path = label_path.replace('jpg','png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 标签由三通道转换为单通道
        label = cv2.imread(label_path, flags=0)
        label[label != 255] = 0
        label[label == 255] = 1
        # 保存npz文件
        np.savez(path2 + s[:-4] + ".npz", image=image, label=label)


npz('D:\ocean big data\REDATA\\train\sentinel_train\image\\', 'D:\ocean big data\REDATA\\train\sentinel_train\mask\\', './datasets/Sen/train_npz/')

npz('D:\ocean big data\REDATA\\test\sentinel\sat\\', 'D:\ocean big data\REDATA\\test\sentinel\gt\\', './datasets/Sen/test_vol_h5/')
