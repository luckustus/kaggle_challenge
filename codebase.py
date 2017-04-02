"""
Created on Sun Apr  2 21:20:43 2017

@authors: Luca Schmidtke, Radu Sibechi, Celâl Kaan Aytemir, Susan Brusselers

The following tutorial was very helpful for some pieces of the code:
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial by Guido Zuidhof

NOTE: This tutorial was not copied! Most of the code has been written by us. 
(especially the segmentaiton, cropping, padding and the neural network implementation)
Some parts (retrieving the dicom data, resampling and rescaling) were looked up in this tutorial
"""

import dicom   # library for reading files in the DICOM format
import numpy as np # calcualtions
import matplotlib.pyplot as plt # plots
import os # directory stuff
import scipy.ndimage
from skimage import measure
import itertools
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.optimizers import SGD
from keras import losses
from keras.utils import to_categorical

# Get the labels for the training data. NOTE: not all junks of code for creating and splitting training/test data and creating
# the training labels are shown because it was performed in many steps. This file provides an overview of the self written functions
# that were used.


stage1 = pd.read_csv('C:/Users/Luca/Desktop/kaggle_challenge/stage1_labels.csv')

labelled_patients = stage1.drop('cancer', axis = 1)

labelled_patients = labelled_patients.values.tolist()

labelled_patients = list(itertools.chain.from_iterable(labelled_patients))


path = 'D:/stage1'
patient_ids = os.listdir(path)


# with this function, we import the DICOM data into python
# slices will be a list of DICOM objects with different parameters

# The slice thickness is missing. This is important because different slice thicknesses can lead to inaccuracies in the
# neural net. The volume data has to be transormed accordingly in order to achieve uniform slice thicknesses. The st
# can be calculated since we know the image positions of different slices along the z-axis.




def get_dicom_data(number):
    patient_data = os.listdir(path + '/' + patient_ids[number]);
    slices =[];
    for s in patient_data:
        slices.append(dicom.read_file(path + '/' + patient_ids[number] + '/' + s));
    
    slices.sort(key = lambda x : float(x.ImagePositionPatient[2]));  # the slices need to be ordered in the z-direction
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2]); #calculate slice thickness
    
    for s in slices:
        s.slice_thickness = slice_thickness
        
    
    
    return slices;
    
# This function extracts the raw pixel information and stores it in a 3D numpy array
# Addtionally, the data is converted back into Hounsfield Units. The necessary parameters are stored
# in the metadata of the DICOM file
    
def get_HU_volume_data(slices):

     Volume = np.zeros((slices[0].pixel_array.shape[0],slices[0].pixel_array.shape[1],len(slices)));
    
     for i in range(len(slices)):
        Volume[:,:,i] = slices[i].pixel_array;
        

     Volume = Volume.astype(np.int16);
         
     outside_image = Volume.min()
     Volume[Volume == outside_image] = 0
           
        
     for i in range(len(slices)):
        Volume[:,:,i] = slices[i].RescaleSlope * Volume[:,:,i] + slices[i].RescaleIntercept; # transform into Hounsfield units
        
    
    
     return Volume;
  

# The data gets resampled accoring to the pixel spacings defined in the DICOM file. The new resolution is
#isotropic (equal in all directions). Do not be confused by the function called 'zoom': It actually does not zoom,
# but rather downsamples the data and interpolates with a spline of third order.

def isotropic_resampling(slices,volume):
    
    pixel_spacing = np.array(slices[0].PixelSpacing + [slices[0].slice_thickness],dtype=np.float32);
    
    new_shape = volume.shape * pixel_spacing;
    
    new_shape = np.round(new_shape);
    
    zoom = new_shape/volume.shape;
    
    volume = scipy.ndimage.interpolation.zoom(volume, zoom, mode='nearest');
    
    return volume;


# Treshold the volume and add ones to the boundaries, in case the patient 'cuts' the air compartment around
# him in two parts and would therefore compromise the following connected component step. You can read more
# about this issue in the preprocessing section of the report  

def threshold(volume):
    
    volume_copy = np.copy(volume)
    
    volume_copy[volume_copy > -320] = 0;
    volume_copy[volume_copy <= -320] = 1;
    
    for i in range (np.shape(volume_copy)[2]):
        
        volume_copy[:,0,i]= 1
        volume_copy[:,np.shape(volume_copy)[1]-1,i] = 1
    
    return volume_copy;
    
    
# lung segmentation. The two biggest connected components found are joint (air around patient and bone/tissue around lungs).
# Conneceted components is run again and the second largest component (the first one is the two previous ones combined)   
# is kept. This corresponds to the lungs. Next, small holes (zeros) within the lungs are filled and then the whole segmentation
# mask is shrunk in order to prevent keeping points at the boundary                 

def lung_segmentation(volume):


    labels = skimage.measure.label(volume,4,0);

    label_air = labels[0,0,0]
    
    uniques,counts = np.unique(labels, return_counts = True);
    
    counts_sorted = np.argsort(counts)
    
    if counts[label_air] != np.amax(counts):
        
        labels[labels == label_air] = counts_sorted[len(counts_sorted)-1]
    
    else:
        
        labels[labels == label_air] = counts_sorted[len(counts_sorted)-2]
        
        
    labels = skimage.measure.label(labels,4,0);
    
    uniques,counts = np.unique(labels, return_counts = True);
    
    counts_sorted = np.argsort(counts)
    
    labels[labels != counts_sorted[len(counts_sorted)-2]] = 0
    
    labels[labels == counts_sorted[len(counts_sorted)-2]] = 1
    
    segmented_lung = (scipy.ndimage.morphology.binary_closing(labels,iterations = 15)).astype(np.int16)
            
    segmented_lung = (scipy.ndimage.morphology.binary_erosion(segmented_lung,iterations = 5)).astype(np.int16)
    
    segmented_lung = volume * segmented_lung
    
    return segmented_lung
    
# The volume still containts a lot of redundant data. This function looks for slices
# that only contain zeros and crops them.
    
def crop_volume(volume):
                
    nonzero = np.ndarray.nonzero(volume)
         
    volume = volume[np.amin(nonzero[0]):np.amax(nonzero[0]),np.amin(nonzero[1]):np.amax(nonzero[1]),np.amin(nonzero[2]):np.amax(nonzero[2])]
           
    return volume
    
# The code here is a bit 'sloppy': The data is zero padded (fill missing pixels with zeros until you reach 350)
# or cropped to a volume of 350x350x350 depending on its input size.
# Not all cases are actually formulated because they simply did not occur.
    
def pad_volume(volume):
    
    
    if np.shape(volume)[0] <= 350 and np.shape(volume)[1] <=350 and np.shape(volume)[2] <= 350:
        
        zero_padded = np.zeros((350,350,350))
        
        lower_x = np.ceil((350-np.shape(volume)[0])/2)
        
        upper_x = np.shape(volume)[0]+np.ceil((350-np.shape(volume)[0])/2)
        
        lower_y = np.ceil((350-np.shape(volume)[1])/2)
        
        upper_y = np.shape(volume)[1]+np.ceil((350-np.shape(volume)[1])/2)
        
        lower_z = np.ceil((350-np.shape(volume)[2])/2)
        
        upper_z = np.shape(volume)[2]+np.ceil((350-np.shape(volume)[2])/2)
        
        zero_padded[lower_x:upper_x,lower_y:upper_y,lower_z:upper_z] = volume
        
        
    elif np.shape(volume)[0] > 350 and np.shape(volume)[1] > 350 and np.shape(volume)[2] <= 350:
        
        zero_padded = np.zeros((350,350,350))
        
        lower_x = np.ceil((np.shape(volume)[0]-350)/2)
        
        upper_x = np.shape(volume)[0] - np.ceil((np.shape(volume)[0]-350)/2)
        
        lower_y = np.ceil((np.shape(volume)[1]-350)/2)
        
        upper_y = np.shape(volume)[1] - np.ceil((np.shape(volume)[1]-350)/2)
        
        lower_z = np.ceil((350-np.shape(volume)[2])/2)
        
        upper_z = np.shape(volume)[2]+np.ceil((350-np.shape(volume)[2])/2)
        
        zero_padded[:,:,lower_z:upper_z] = volume[lower_x:upper_x,lower_y:upper_y,:]
    
    elif np.shape(volume)[0] <= 350 and np.shape(volume)[1] <= 350 and np.shape(volume)[2] > 350: 
        
         zero_padded = np.zeros((350,350,350))
        
         lower_x = np.ceil((350-np.shape(volume)[0])/2)
        
         upper_x = np.shape(volume)[0]+np.ceil((350-np.shape(volume)[0])/2)
         
         lower_y = np.ceil((350-np.shape(volume)[1])/2)
         
         upper_y = np.shape(volume)[1]+np.ceil((350-np.shape(volume)[1])/2)
         
         lower_z = np.ceil((np.shape(volume)[2]-350)/2)
         
         upper_z = np.shape(volume)[2] - np.ceil((np.shape(volume)[2]-350)/2)
         
         zero_padded[lower_x:upper_x,lower_y:upper_y,:] = volume[:,:,lower_z:upper_z]
         
    elif np.shape(volume)[0] <= 350 and np.shape(volume)[1] > 350 and np.shape(volume)[2] <= 350:
        
         zero_padded = np.zeros((350,350,350))
         
         lower_x = np.ceil((350-np.shape(volume)[0])/2)
        
         upper_x = np.shape(volume)[0]+np.ceil((350-np.shape(volume)[0])/2)
         
         lower_y = np.ceil((np.shape(volume)[1]-350)/2)
        
         upper_y = np.shape(volume)[1] - np.ceil((np.shape(volume)[1]-350)/2)
         
         lower_z = np.ceil((350-np.shape(volume)[2])/2)
        
         upper_z = np.shape(volume)[2]+np.ceil((350-np.shape(volume)[2])/2)
         
         zero_padded[lower_x:upper_x,:,lower_z:upper_z] = volume[:,lower_y:upper_y,:]
         
         
         return zero_padded
        
# In the final step, the data is downsampled and interpolated to 128x128x128, tresholded again to remove
# air/lung components and rescaled between 0 and 1
def final(volume):
    
    final = (scipy.ndimage.interpolation.zoom(volume,128/350,mode = 'nearest')).astype(np.float32)
    
    MIN_BOUND = -4000.0
    
    MAX_BOUND = 400.0
    
    volume[volume < -320] = 0
    
    final = ((final - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)).astype(np.float32)
    final[final>1] = 1.
    final[final<0] = 0.




###########################################################################

# ARCHITECTURE OF THE NETWORK IMPLEMENTED IN KERAS

# input data ('array') is a 5D array with dimensions (volume_x, volume_y, volume_z, number of samples, number of channels (=1))

# 'training_labels' is array with dimensions (number of samples, 2) and has labels 1 (cancer) and 0 (non-cancer)
       
model = Sequential()

model.add(Conv3D(8, (5, 5, 5), strides = (2,2,2), activation='relu', input_shape=(128, 128, 128, 1)))

model.add(Conv3D(16, (3, 3, 3), strides = (2,2,2),activation='relu'))

model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Conv3D(32, (3, 3, 3), strides = (2,2,2),activation='relu'))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dense(128,activation='relu'))

model.add(Dense(2,activation='softmax'))

sgd = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy', optimizer = sgd, metrics=['accuracy'] )

history = model.fit(array, training_labels, epochs=10, batch_size = 1)  # train model


# plot accuracy and loss during training
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
   
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
