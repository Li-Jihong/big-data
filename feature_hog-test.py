import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from cv2 import HOGDescriptor, imread, resize, COLOR_BGR2GRAY, cvtColor

data_dir = "RawDataset"

def _get_label(pic_name):
    set_str = pic_name.strip("Locate{}.jpg")  # cut paddings
    label = set_str[-set_str[::-1].index(","):] # get label after the last ','
    return int(label)-1

# OpenCV HOG Descriptor parameters
winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                    histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

def _get_pic_data(dir_name):
    pic_names = os.listdir(dir_name)
    img_arrs, labels = [], []
    for pic_name in pic_names:
        img = imread(dir_name + "/" + pic_name)   # OpenCV reads the image
        img = cvtColor(img, COLOR_BGR2GRAY)       # Convert to grayscale
        img = resize(img, winSize)                # Resize image to match HOG Descriptor size
        hog_features = hog.compute(img)           # Compute HOG features
        label = _get_label(pic_name)              # Get the label of the image
        img_arrs.append(hog_features.flatten())  # Flatten the features
        labels.append(label)
    return img_arrs, labels

def load_raw():
    """获取特征数据集，x是feature，y是label"""
    x, y = _get_pic_data(data_dir)
    x = np.array(x)
    y = np.array(y)
    return x, y

x, y = load_raw()
print("load data successful")

x_train, x_test, y_train, y_test = train_test_split(x, y)  # 使用 sklearn 进行划分

# 连接数据集
train_df = np.concatenate((x_train, np.expand_dims(y_train, 1)), axis=1)
test_df = np.concatenate((x_test, np.expand_dims(y_test, 1)), axis=1)

# 写成 csv 文件
np.savetxt("src/train.csv", train_df, delimiter=",", fmt="%.1f")
np.savetxt("src/test.csv", test_df, delimiter=",", fmt="%.1f")

print("data saved successful")
