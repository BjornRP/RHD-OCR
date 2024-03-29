# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:29:50 2019

@author: bpe043
"""

import os

# Function to return list over full-path image names
def listdir_fullpath(folder):
    return [os.path.join(folder, image) for image in os.listdir(folder)]

# When images are in separate folders
def return_image_list_folders(root):
    
    image_list = []
        
    for subdir, dirs, files in os.walk(root):
        for d in dirs:
            l = listdir_fullpath(subdir + d)
            image_list.extend(l)

    return image_list       
            

# When images are in separate folders, and you need a list of labels
# Takes as input the root directory of the image sub-folders
def return_image_list_labels_folders(root):
    
    image_list = []
    label_list = []
    
    # Used as a start/stopping point for slicing, when inserting labels based on how many images from each class there is
    init = 0
            
    for subdir, dirs, files in os.walk(root):
        
        # This for-loop iterates over all subfolders, uses the listdir_fullpath function to get the image's full path added to the list of images
        # It also fills in the label list with labels according to the number of images
        for d in dirs:
            l = listdir_fullpath(subdir + d)
            image_list.extend(l)            
            label_list[init:] = [ int(d) for i in range( len(l) )]
            
            init += len(l)


    return image_list, label_list
        