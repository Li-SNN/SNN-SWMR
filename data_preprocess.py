# -*- coding: utf-8 -*-
"""
@Author: Pangpd (https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: Li-SNN，
"""
import math
import os
import random
from math import ceil

import scipy.io as sio
import scipy.ndimage
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.hyper_pytorch import HyperData


def loadData(data_path, name, num_components): 
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        label_names = ["Alfalfa", "Corn-notill", "Corn-mintill",
                       "Corn", "Grass-pasture", "Grass-trees",
                       "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                       "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                       "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                       "Stone-Steel-Towers"]
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        label_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees',
                       'Painted metal sheets', 'Bare Soil', 'Bitumen',
                       'Self-Blocking Bricks', 'Shadows']
    elif name == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
        label_names = ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
                       "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                       "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk ",
                       "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained",
                       "Vinyard_vertical_trellis"]
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        label_names = ["Scrub", "Willow swamp", "CP hammock",
                       "Slash pine", "Oak/Broadleaf ", "Hardwood",
                       "Swamp", "Graminoid marsh", "Spartina marsh",
                       "Cattail marsh", "Salt marsh", "Mud flats", "Water"]
    elif name == 'WHHC':
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
        label_names = ["Strawberry", "Cowpea",
                       "Soybeam", "Sorghum ", "Water spinach",
                       "Watermelon", "Greens", "Trees","Grass","Red roof","Gray roof",
                       "Plastic","Bare soil","Road","Bright object","Water"]
    elif name == 'WHLK':
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
        label_names = ["Corn", "Cotton",
                       "Sesame", "Broad-leaf soybean", "Narrow-leaf soybean",
                       "Rice", "Water", "Roads and houses", "Mixed weed"]
    elif name == 'WHHH':
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
        label_names = ["Red roof", "Road",
                       "Bare soil", "Cotton ", "Cotton firewood",
                       "Rape", "Chinese cabbage", "Pakchoi", "Cabbage", "Tuber mustard", "Brassica parachinensis",
                       "Brassica chinensis", "Small Brassica chinensis","Lactuca sativa", "Celtuce", "Film covered lettuce", "Romaine lettuce",
                       "Carrot", "White radish", "Garlic sprout", "Broad bean","Tree"]
    elif name == 'HU':
        data = sio.loadmat(os.path.join(data_path, '2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat'))['ans']
        labels = sio.loadmat(os.path.join(data_path, 'GRSS2013.mat'))['name']
        label_names = ["Healthy grass", "Stressed grass",
                       "Synthetic grass", "Trees ", "Soil",
                       "Water", "Residential", "Commercial", "Road", "Highway", "BRailway",
                       "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"]
    else:
        print("NO DATASET")
        exit()

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    if num_components != None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    # data = MinMaxScaler().fit_transform(data)
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels)) - 1
    return data, labels, num_class, label_names


def load_hyper(data_path, name, spatial_size, use_val=True, n_samples=0.15, train_percent=0.023,
               batch_size=64, components=None, rand_state=None):
    print('进入data_preprocess.py中的load_hyper方法！')
    print("调用loadData方法，传入的参数：", data_path, name, components)
    data, labels, num_classes, _ = loadData(data_path, name, components)
    print("batch_size:",batch_size)
    print("spatial_size:", spatial_size)
    print("调用loadData方法，返回的高光谱图像的地物种类数量：", num_classes)
    pixels, labels = createImageCubes(data, labels, windowSize=spatial_size, removeZeroLabels=True)
    #print("pixels,labels:", pixels,labels)
    bands = pixels.shape[-1]
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_threshold_random(pixels, labels, n_samples,
                                                                                 train_percent, rand_state=rand_state)
    del pixels, labels
    train_hyper = HyperData((np.transpose(X_train, (0, 3, 1, 2)).astype("float32"), y_train),None)
    test_hyper = HyperData((np.transpose(X_test, (0, 3, 1, 2)).astype("float32"), y_test),None)
    if use_val:
        val_hyper = HyperData((np.transpose(X_val, (0, 3, 1, 2)).astype("float32"), y_val), None)
    else:
        val_hyper = None
    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=batch_size,shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader, num_classes, bands

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    num_labels = np.count_nonzero(y[:, :])  # 标签样本总数, 非0数
    print("num_labels(标签样本总数, 非0数):", num_labels)
    margin = int((windowSize - 1) / 2)
    print("margin::", margin)
    print("调用padWithZeros前的X.shape：", X.shape)
    zeroPaddedX = padWithZeros(X, margin=margin)
    print("调用padWithZeros后的zeroPaddedX.shape：", zeroPaddedX.shape)
    patchIndex = 0
    if removeZeroLabels == True:
        patchesData = np.zeros((num_labels, windowSize, windowSize, X.shape[2]), dtype='float32')
        patchesLabels = np.zeros(num_labels)

        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                if y[r - margin, c - margin] > 0:
                    patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                    patchesData[patchIndex, :, :, :] = patch
                    patchesLabels[patchIndex] = y[r - margin, c - margin]
                    patchIndex = patchIndex + 1

    if removeZeroLabels == False: 
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype="float32")
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                patchIndex = patchIndex + 1
    patchesLabels -= 1

    return patchesData, patchesLabels.astype("int")

def split_data_threshold_random(pixels, labels, n_samples, train_percent, rand_state=None):
    train_set_size = []
    for cl in np.unique(labels):
        pixels_cl = len(pixels[labels == cl]) 
        pixels_cl = round(pixels_cl * train_percent)
        # if(pixels_cl < 1000):

        print("训练集占比",train_percent)
        print("样本",cl,"数量：",pixels_cl)
        # pixels_cl = min(ceil(pixels_cl * 0.3), n_samples) 
        # if pixels_cl < n_samples:
        #     pixels_cl = ceil(pixels_cl * 0.8)
        # else:
        #     pixels_cl = n_samples
        train_set_size.append(pixels_cl)  
    pixels_number = np.unique(labels, return_counts=1)[1] 
    tr_size = int(sum(train_set_size))
    print("总数量：",tr_size)
    te_size = int(sum(pixels_number)) - int(sum(train_set_size))
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])
    train_x = np.empty((sizetr))
    train_y = np.empty((tr_size), dtype=int)
    X_test = np.empty((sizete))
    y_test = np.empty((te_size), dtype=int)
    trcont = 0;
    tecont = 0;
    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                train_x[trcont, :, :, :] = a
                train_y[trcont] = b
                trcont += 1
            else:
                X_test[tecont, :, :, :] = a
                y_test[tecont] = b
                tecont += 1
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=train_percent, stratify=train_y,
                                                      random_state=rand_state)
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_data_percent(pixels, labels, train_samples, val_samples, rand_state=None):
    train_set_size = []  
    for cl in np.unique(labels):
        pixels_cl = len(pixels[labels == cl])  
        train_pixels_cl = min(ceil(pixels_cl * 0.3), train_samples)  
        train_set_size.append(train_pixels_cl) 

    val_set_size = [ceil(i * val_samples) for i in train_set_size] 

    pixels_number = np.unique(labels, return_counts=1)[1]  

    tr_size = int(sum(train_set_size))
    val_size = int(sum(val_set_size))
    te_size = int(sum(pixels_number)) - tr_size - val_size
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizeval = np.array([val_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])

    X_train = np.empty((sizetr))
    y_train = np.empty((tr_size), dtype=int)
    X_val = np.empty((sizeval))
    y_val = np.empty((val_size), dtype=int)
    X_test = np.empty((sizete))
    y_test = np.empty((te_size), dtype=int)
    trcont = 0;
    valcont = 0;
    tecont = 0;

    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                X_train[trcont, :, :, :] = a
                y_train[trcont] = b
                trcont += 1
            elif cont < train_set_size[cl] + val_set_size[cl]:
                X_val[valcont, :, :, :] = a
                y_val[valcont] = b
                valcont += 1
            else:
                X_test[tecont, :, :, :] = a
                y_test[tecont] = b
                tecont += 1

    X_train, y_train = random_unison(X_train, y_train, rstate=rand_state)
    # X_test, y_test = random_unison(X_test, y_test, rstate=rand_state)
    X_val, y_val = random_unison(X_val, y_val, rstate=rand_state)
    return X_train, y_train, X_val, y_val, X_test, y_test


def random_unison(a, b, rstate=None):
    assert len(a) == len(b) 
    p = np.random.RandomState(seed=rstate).permutation(len(a))  
    return a[p], b[p]
