#====================================================================
# Producing attention vizualizations for the conv net. 
# This code will produce visualizations based off model input pictures 
# and a model h5 file. 
#====================================================================


#====================================================================
# for visualizing network
from vis.utils import utils
from vis.visualization import overlay
from keras import activations
from keras.utils.visualize_util import plot
import cv2

from vis.utils import utils  
from keras.models import load_model

import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam

# for plot 
from vis.utils import utils
from matplotlib import pyplot as plt
from vis.visualization import visualize_cam

import csv 
import os
#====================================================================


#====================================================================
# load the images
# this is using the example training images, to save space. 
# you can use it on the recorded images you have from an automomous run
lines = os.listdir('./data/IMG')

# load the model
model = load_model('./model.h5') 

# cut short the image list if wanted
lines_cut = lines

# creates an image with attention overlay for each picture fed in. 
# change 'small_values' to None for right turn, "negate" for left turn, or leave for maintain straight
def attention_overlay(lines, out_loc='./attention/', model = model):
    for line in lines:
        #source_path = line
        # split character modified because image data was collected on windows 10
        #filename = source_path.split('\\')[-1]
        filename = line
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        crop_img = image[65:140,0:320,:]
        #masked_image = image_mask(crop_img)
        grads = visualize_cam(model, layer_idx=-1, filter_indices=0, seed_input=crop_img, grad_modifier = 'small_values')        
        #heatmap = np.uint8(cm.jet(grads))
        out_image = overlay(grads, crop_img, alpha = 0.4)
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        file_path = out_loc + filename
        cv2.imwrite(file_path, out_image)
        del image
        del crop_img
        del out_image
    return 0
    
# run the attention function   
attention_overlay(lines_cut)
#====================================================================
   

