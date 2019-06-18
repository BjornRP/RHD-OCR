# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:26:30 2019

@author: bpe043
"""
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import pickle

# This is to facilitate importing my own helper functions
import sys
sys.path.insert(0, '//homer.uit.no/bpe043/Desktop/Test_Projects/HelperFunctions')

from loadImagesFromFolders import return_image_list_labels_folders



#Load model and create data generator
model = load_model('BalancedTraining_1000Validation_sparseCategoricalCrossentropy_ESValAcc_LongPatience_SimpleTrainGen_Model.HDF5')
validation_datagen = ImageDataGenerator(rescale=1./255)

#Get validation images
root_dir = 'input/'

list_of_images, list_of_labels = return_image_list_labels_folders(root_dir)

# Make a prediction for each image in List of images:
list_of_predictions = []    # Holds the predicted label for each image, used later

for image in list_of_images :
    img = cv2.imread(image, -1)
    
    #Reshape image for prediction
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    
     # Prediction
    for batch in validation_datagen.flow(img, batch_size=1):
        pred = model.predict(batch)
        
        predictions = pred.reshape(pred.shape[1])
        
        # Find the confidence of the max value, and it's index
        max_value = max(predictions)
        max_index = np.where(predictions == np.amax(predictions))[0][0]
        list_of_predictions.append(max_index)
        
        break       # Exit after the prediction has been made, and move to the next image
        
# Recall and Precision with bonus F1 score
with open('labels.txt', 'wb') as fp:
    pickle.dump(list_of_labels, fp)

with open('predictions.txt', 'wb') as fp:
    pickle.dump(list_of_predictions, fp)
        