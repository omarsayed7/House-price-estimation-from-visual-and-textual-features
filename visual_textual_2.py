#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:26:55 2019

@author: omar
"""
import cv2
import glob
import time
import locale
import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt 

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.layers.core import Dropout
from keras.callbacks import TensorBoard
from keras.models import model_from_yaml
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization



Model_Name = "Final_House_price_estimation- {}".format(int(time.time()))
Model_Name_yml = "Final_House_price_estimation- {}.yaml".format(int(time.time()))
tensorboard = TensorBoard(log_dir= 'logs/{}'.format(Model_Name))

def textual_data_cleaning(data_path):
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    dataset = pd.read_csv(data_path,sep = ' ', names = cols)
    dataset = dataset.dropna()
    return dataset
    
    
    
    
    
def textual_data_preprocessing(cleaned_data):
    #categorcal data encoding 
    cleaned_data = pd.get_dummies(cleaned_data, columns = ["zipcode"])
    
    textual_train, textual_test= train_test_split(cleaned_data,test_size = 0.15, random_state = 42)
    #split the features and labels from the data
    max_price = textual_train["price"].max()
    y_train = textual_train.iloc[:,3].values
    y_test = textual_test.iloc[:,3].values
    y_test_actual = y_test
    textual_train = textual_train.drop(["price"],axis = 1)
    textual_test = textual_test.drop(["price"],axis = 1)
    X_train = textual_train.iloc[:,:].values
    X_test  = textual_test.iloc[:,:].values
    #Feature Scaling 
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    return  textual_train, textual_test , X_train , X_test , y_train , y_test , max_price , y_test_actual

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

def Prepare_Final_Data(data_path):
    cleaned_data = textual_data_cleaning(data_path)
    textual_train, textual_test , X_train , X_test , y_train , y_test , max_price , y_test_actual = textual_data_preprocessing(cleaned_data)
    train_images , test_images = Visual_datapreprocessing(textual_train,textual_test)
    train_images = train_images /255.0
    test_images  = test_images /255.0
    y_train = y_train / max_price
    y_test = y_test / max_price
    
    return X_train , X_test , y_train , y_test , train_images, test_images , y_test_actual
    
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
    
    model.add(Conv2D(64, (3,3), activation= 'elu', padding='same'))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, (3,3), activation= 'elu', padding='same'))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #Fully connected NN
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization(axis= -1))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation= 'relu'))
    
    return model





X_train , X_test , y_train , y_test , train_images, test_images, y_test_actual = Prepare_Final_Data("Houses Dataset/HousesInfo.txt")

Dense_NN = Dense_Network(X_train.shape[1])
CNN = Conv_Network()

Multi_Input = concatenate([Dense_NN.output, CNN.output])

Final_Fully_Connected_Network = Dense(4, activation = 'relu')(Multi_Input)
Final_Fully_Connected_Network = Dense(1)(Final_Fully_Connected_Network)

model = Model(inputs = [Dense_NN.input , CNN.input], outputs = Final_Fully_Connected_Network)

model.compile(optimizer = 'adam', loss = 'mse')

model.fit([X_train,train_images], y_train, validation_split = 0.15,
		epochs = 50,
		batch_size = 8, 
		callbacks= [tensorboard])



preds = model.predict([X_test,test_images])
error = preds.flatten() - y_test
squared_error = error ** 2
MSE = np.mean(squared_error)

'''
R2 score 
'''
r2_score_test = r2_score(y_test,preds.flatten())


# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - y_test
percentDiff = (diff / y_test) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
print("MSE = ", MSE)
print("R2_score = ", r2_score_test)


# serialize model to YAML
model_yaml = model.to_yaml()
with open(Model_Name_yml, "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
