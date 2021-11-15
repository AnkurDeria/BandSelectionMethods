from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
import pprint
from PIL import Image
import os as os



def loadData(dataset, name):
        data_path = "../MomentumSSRN/"
        if name == "Houston":
            if dataset == 'HSITest' or dataset == 'HSITrain':
                data = sio.loadmat('./../Houston/houston.mat')['houston']
            elif dataset == 'LIDARTest' or dataset == 'LIDARTrain':
                data = np.array(Image.open('./../Houston/houston_lidar.tif'))
                data = data.reshape(data.shape[0],data.shape[1],1)

            if dataset == 'HSITest' or dataset == 'LIDARTest':
                labels = sio.loadmat('./../Houston/houston_gt.mat')['houston_gt_te']
            elif dataset == 'HSITrain' or dataset == 'LIDARTrain':
                labels = sio.loadmat('./../Houston/houston_gt.mat')['houston_gt_tr']
            return data, labels

        elif(name == "Trento"):
            if dataset == 'HSITest' or dataset == 'HSITrain':
                data = sio.loadmat('./../Trento/HSI.mat')['HSI']
            elif dataset == 'LIDARTest' or dataset == 'LIDARTrain':
                data = sio.loadmat('./../Trento/LiDAR.mat')['LiDAR']

            if dataset == 'HSITest' or dataset == 'LIDARTest':
                labels = sio.loadmat('./../Trento/TSLabel.mat')['TSLabel']
            elif dataset == 'HSITrain' or dataset == 'LIDARTrain':
                labels = sio.loadmat('./../Trento/TRLabel.mat')['TRLabel']
            return data, labels
        elif(name == "KSC"):
            if dataset == 'HSITest' or dataset == 'HSITrain':
                data = sio.loadmat('./../KSC/KSC_corrected.mat')['KSC']

            if dataset == 'HSITest' or dataset == 'LIDARTest':
                labels = sio.loadmat('./../KSC/LTeC_up.mat')['LTeC']
            elif dataset == 'HSITrain' or dataset == 'LIDARTrain':
                labels = sio.loadmat('./../KSC/LTrC_up.mat')['LTrC']

            return data, labels
    
        elif(name == "MUUFL"):
            allData = data = sio.loadmat('./../MUUFL/muufl_share.mat')

            dataHSI = allData['hsi_img']

            dataLIDAR = allData['lidarz']

            labels = allData['labels']
            return dataHSI, dataLIDAR, labels
        elif name == "MUUFLSR":
            allData = data = sio.loadmat('./../MUUFL/muufl_share.mat')

            dataHSI = allData['hsi_img']

            dataLIDAR = allData['rgb']

            labels = allData['labels']
            
            return dataHSI, dataLIDAR, labels
       
        elif name == 'IP':
            data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
            labels = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected_gt.mat'))['indian_pines_gt']
            return data, labels
        elif name == 'SA':
            data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
            labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
            return data, labels
        elif name == 'UP':
            data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
            labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
            return data, labels



def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def normalizeHSI(X):
    for i in range(X.shape[2]):
        minimal = X[:, :, i].min()
        maximal = X[:, :, i].max()
        X[:, :, i] = (X[:, :, i] - minimal)/(maximal - minimal)
    return X
        
def normalizeLIDAR(X):
    minimal = X.min()
    maximal = X.max()
    X = (X - minimal)/(maximal - minimal)
    return X

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    print(zeroPaddedX.shape)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=int)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        
    return patchesData, patchesLabels

def CreateDataset(name ="",windowSize = 0, HSI_Only= True):
    X, y = loadData('HSITrain',name)

    print("X shape = ", X.shape,"\nY shape = ", y.shape)
    K = X.shape[2]
    X1 = normalizeHSI(X)
    if windowSize > 0:
        X1, yPatch = createImageCubes(X1, y, windowSize=windowSize)
    else:
        yPatch = y

    print("X1 shape after IMAGE CUBES = ", X1.shape,"\nY shape after IMAGE CUBES = ", yPatch.shape)


    HSI_Train = X1
    Train_Labels = yPatch
    
    X, y = loadData('HSITest',name)

    
    K = X.shape[2]
    X2 = normalizeHSI(X)
    print("X shape = ", X2.shape,"\nY shape = ", y.shape)
    if windowSize > 0:
        X2, yPatch = createImageCubes(X2, y, windowSize=windowSize)
    else:
        yPatch = y
    

    print("X1 shape after IMAGE CUBES = ", X2.shape,"\nY shape after IMAGE CUBES = ", yPatch.shape)

    HSI_Test = X2
    Test_Labels = yPatch
    if not HSI_Only:
        X, y = loadData('LIDARTrain',name)
        if len(X.shape) < 3:
            X = X.reshape((X.shape[0],X.shape[1],1))
        print("X shape = ", X.shape,"\nY shape = ", y.shape)
        if X.shape[2] > 1:
            X = normalizeHSI(X)
        else:
            X = normalizeLIDAR(X)
        X, y = createImageCubes(X, y, windowSize=windowSize)

        print("X shape after IMAGE CUBES = ", X.shape,"\nY shape after IMAGE CUBES = ", y.shape)

        LIDAR_Train = X
        
        X, y = loadData('LIDARTest',name)
        if len(X.shape) < 3:
            X = X.reshape((X.shape[0],X.shape[1],1))
        print("X shape = ", X.shape,"\nY shape = ", y.shape)
        if X.shape[2] > 1:
            X = normalizeHSI(X)
        else:
            X = normalizeLIDAR(X)
        X, y = createImageCubes(X, y, windowSize=windowSize)

        print("X shape after IMAGE CUBES = ", X.shape,"\nY shape after IMAGE CUBES = ", y.shape)

        LIDAR_Test = X
        return HSI_Train,HSI_Test,LIDAR_Train,LIDAR_Test,Train_Labels-1,Test_Labels-1
    else:
        return HSI_Train,HSI_Test,Train_Labels-1,Test_Labels-1



