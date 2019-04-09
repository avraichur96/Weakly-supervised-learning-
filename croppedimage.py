import numpy as np 
import cv2
from os import listdir 
import pandas as pd 

""" new dataframe to store the cropped images"""

cropped_train = pd.DataFrame()

directory = "C:\\Users\\LENOVO\\task\\data"    """ replace with required path"""
x, y = 0 , 0


""" loop through the imagees in directory k
perform the sliding window operation and append to the cropped_train dataframe"""

for name in listdir(directory):
    path = "C:\\Users\\LENOVO\\task\\data"    """ replace with required path"""
    path = path + '\\' + name 
    nimg = cv2.imread(path)
    rimg = cv2.resize(nimg,(500,500),interpolation=cv2.INTER_AREA)
    y = 0
    for p in range(4):
        for j in range(4):
            img = rimg[y:y+224, x:x+224, 0]
            img = np.reshape(img,(50176,))
            yo = pd.Series(img) 
            cropped_train = cropped_train.append(yo, ignore_index = True )
            x = x + 64
        y = y + 64
        x=0 
        


cropped_train.to_csv("cropped.csv")
