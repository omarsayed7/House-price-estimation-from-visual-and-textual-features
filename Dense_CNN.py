#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:53:39 2019

@author: omar
"""
import numpy as np 
import pandas as pd 
import glob 
import cv2 
import os 
import time

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate


Model_Name = "Final_House_price_estimation- {}".format(int(time.time()))
tensorboard = TensorBoard(log_dir= 'logs/{}'.format(Model_Name))

np.random.seed(7)

def Textual_data_Pre_Processing(input_path):
	#initialize the list of column names in csv file
	columns = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	data_frame = pd.read_csv(input_path, sep = " ", header = None, names= columns)

	#determine the unique zipcodes and the number of data with each unique zipcode
	zipcodes = data_frame["zipcode"].value_counts().keys().tolist()
	counts = data_frame["zipcode"].value_counts().tolist()

	#loog over each unique zipcodes and their counts

	for (zipcode, count) in zip(zipcodes,counts):
		#the zipcode counts is unbalnced so let's remove any houses with less than 
		#25 houses per zip code 

		if count < 25 :
			indxs = data_frame[data_frame["zipcode"] == zipcode ].index
			data_frame.drop(indxs, inplace = True)

	return data_frame
	


def Textual_data_processing(dataset,TRAIN_X, TEST_X):
	dataset = dataset
	cols = ["bedrooms", "bathrooms","area"]

	cs = MinMaxScaler()
	train_Scaling = cs.fit_transform(TRAIN_X[cols])
	test_Scaling = cs.transform(TEST_X[cols])

	zipcode_Binarizer = LabelBinarizer()
	zipcode_Binarizer.fit(dataset["zipcode"])
	train_Binarizer = zipcode_Binarizer.transform(TRAIN_X["zipcode"])
	test_Binarizer = zipcode_Binarizer.transform(TEST_X["zipcode"])

	X_train = np.hstack([train_Binarizer,train_Scaling])
	X_test = np.hstack([test_Binarizer,test_Scaling])

	return X_train, X_test



def Visual_data_processing(dataset):
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
        housePaths = sorted(list(glob.glob("{}_*".format(i+1))))
        
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

def END_TO_END_NETWORK():
	data_path = "HousesInfo.txt"
	dataset = Textual_data_Pre_Processing(data_path)
	images = Visual_data_processing(dataset)
	images /= 255.0

	(train_Textual, test_Textual, train_Images,test_Images) = train_test_split(dataset, images, test_size=0.25, random_state=42)
	#train_Texture = (features and price(Y)) so a test_Texture ,, train_Images = only images without labels (Y)
	X_train, X_test = Textual_data_processing(dataset,train_Textual, Test_Textual)
	train_Y = train_Textual["price"]
	test_Y = test_Textual["price"]

	Dense_Network = Dense_Network(train_Textual.shape[1])
	CNN = Conv_Network()

	Multi_Input = concatenate([Dense_Network.output, CNN.output])

	Final_Fully_Connected_Network = Dense(4, activation = 'relu')(Multi_Input)
	Final_Fully_Connected_Network = Dense(1)(Final_Fully_Connected_Network)

	model = Model(inputs = [Dense_Network.input , CNN.input], outputs = Final_Fully_Connected_Network)

	model.compile(optimizer = 'adam', loss = losses.mean_squared_error)

	model.fit([X_train,train_Images], train_Y, validation_data = ([X_test, test_Images], test_Y),
		epochs = 10,
		batch_size = 8, 
		callbacks= [tensorboard])

	preds = model.predict([test_Textual,test_Images])

	error = preds - test_Y
	squared_error = error ** 2
	MSE = np.mean(squared_error)

	print("MSE = ", MSE)
	keras.losses.mean_squared_error(test_Y, preds)
	print("KERAS_MSE = ", MSE)
	return preds , test_Y


preds , test_Y = END_TO_END_NETWORK()




