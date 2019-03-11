#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:26:55 2019

@author: omar
"""
import cv2
import glob
import time
import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.layers.core import Dropout
from keras.callbacks import TensorBoard
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization


Model_Name = "Final_House_price_estimation- {}".format(int(time.time()))
tensorboard = TensorBoard(log_dir= 'logs/{}'.format(Model_Name))

def textual_data_cleaning(data_path):
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    dataset = pd.read_csv(data_path,sep = ' ', names = cols)
    dataset = dataset.dropna()
    return dataset
    
    
    
    
    
def textual_data_preprocessing(cleaned_data):
    #categorcal data encoding 
    cleaned_data = pd.get_dummies(cleaned_data, columns = ["zipcode"])
    
    textual_train, textual_test= train_test_split(cleaned_data,test_size = 0.2, random_state = 42)
    #split the features and labels from the data
    y_train = textual_train.iloc[:,3].values
    y_test = textual_test.iloc[:,3].values
    textual_train = textual_train.drop(["price"],axis = 1)
    textual_test = textual_test.drop(["price"],axis = 1)
    X_train = textual_train.iloc[:,:].values
    X_test  = textual_test.iloc[:,:].values
    #Feature Scaling 
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    return  textual_train, textual_test , X_train , X_test , y_train , y_test

def Genrate_Visual_images(dataset):
    '''
    Create a montage that combines/tiles all four images into 
    a single image and then pass the montage through the CNN
    #######################################################
    the final image contains :
        The bathroom image in the top-left
        The bedroom image in the top-right
        The frontal view in the bottom-right
        The kitchen in the bottom-left
        
    '''
    images = []
    
    for i in dataset.index.values:
        housePaths = sorted(list(glob.glob("Houses Dataset/{}_*".format(i+1))))
        
        inputImages = []
        outputImage = np.zeros((64,64,3), dtype = "uint8")
                
        for housePath in housePaths : 
            image = cv2.imread(housePath)
            image = cv2.resize(image,(32,32))
            inputImages.append(image)
            
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]
            
        images.append(outputImage)
        
    return np.array(images)

def Visual_datapreprocessing(X_train,X_test):
    train_images = Genrate_Visual_images(X_train)
    test_images = Genrate_Visual_images(X_test)
    
    return train_images , test_images
    
def Dense_Network(dim):
	model = Sequential()
	model.add(Dense(8, input_dim = dim, activation = 'relu'))
	model.add(Dense(4,activation = 'relu'))

	return model

def Conv_Network():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation= 'relu', padding='same', input_shape = (64,64,3)))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3,3), activation= 'tanh', padding='same'))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3,3), activation= 'tanh', padding='same'))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #Fully connected NN
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization(axis= -1))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation= 'relu'))
    
    return model



cleaned_data = textual_data_cleaning("Houses Dataset/HousesInfo.txt")
textual_train, textual_test , X_train , X_test , y_train , y_test = textual_data_preprocessing(cleaned_data)
train_images , test_images = Visual_datapreprocessing(textual_train,textual_test)
train_images = train_images /255.0
test_images  = test_images /255.0

Dense_NN = Dense_Network(X_train.shape[1])
CNN = Conv_Network()

Multi_Input = concatenate([Dense_NN.output, CNN.output])

Final_Fully_Connected_Network = Dense(4, activation = 'relu')(Multi_Input)
Final_Fully_Connected_Network = Dense(1)(Final_Fully_Connected_Network)

model = Model(inputs = [Dense_NN.input , CNN.input], outputs = Final_Fully_Connected_Network)

model.compile(optimizer = 'adam', loss = 'mse')

model.fit([X_train,train_images], y_train, validation_data = ([X_test, test_images], y_test),
		epochs = 100,
		batch_size = 4, 
		callbacks= [tensorboard])

preds = model.predict([X_test,test_images])

error = preds.flatten() - y_test
squared_error = error ** 2
MSE = np.mean(squared_error)

print("MSE = ", MSE)
