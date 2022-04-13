# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:50:14 2022

Kacper Dudek - 18228798
Christian Ryan - 18257356
Charlie Gorey O'Neill - 18222803
Sean McTiernan - 18224385

description: looks intoa folder named 'roi' and checks all of the png files, 
it then increasses the contrast resizes them to 64x64 pixels and flattetns into 
a 4096d array, then it searches for the closest vector form the descriptors file
and assigns a label to a the roi image
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as transform

from os import listdir

from PIL import Image,ImageEnhance
from scipy import spatial

def classifier(folder,file):
    #create a path string
    path = f'{folder}/{file}'
    
    #open image
    im = Image.open(path)
    
    #load in descriptor vectors
    descriptors = np.load("1-NN-descriptor-vects.npy")
    
    """DEBUG-show original image
    plt.imshow(im)
    plt.show()
    #END_DEBUG"""
    
    #increase image contrast
    enhancer=ImageEnhance.Contrast(im)
    newIm = enhancer.enhance(5)
    
    """DEBUG-show high contrast image
    plt.imshow(newIm)
    plt.show()
    #END_DEBUG"""
    
    #convert image to greyscale
    greyIm = newIm.convert('L')
    
    #resize the image
    resizedIm = ResizeImage(greyIm)
    vector = Vectorisation(resizedIm)
    
    #find closest vector in descriptors
    label = Search(descriptors,vector,file)
    
    if label!="REJECTED":
        #plot the image with a title corresponding to the file name
        title = "File name: " + file + " Label: " + label
        plt.imshow(resizedIm,cmap='gray')
        plt.title(title)
        plt.show()

#change the image to 64x64
def ResizeImage(im):
    #shrink the image to 64x64 pixels
    newIm = transform.resize(np.asarray(im),(64,64))
    
    """DEBUG-show the image after resize
    plt.imshow(newIm,cmap='gray')
    plt.show()
    #END_DEBUG"""
    
    return newIm
    
#transform the numpy array into a 4096-dimensional vector
def Vectorisation(im):
    #scale the elements of the image array to fit between 0 and 255
    newIm = (im*255)
    
    #take an average of the matrix
    mean = np.average(newIm)
    
    #subtract the average to make the image have an average of 0
    newIm -= mean
    
    #flatten the numpy array to a 4096-dimesnional vector
    vector = newIm.flatten()
    
    #normalise the vector
    unitVector = vector / np.linalg.norm(vector) 
    
    return unitVector
    
#find nearest vector from descriptors array
def Search(descriptors,vector,file):
    #create a tree of descriptors minus the labels
    tree = spatial.KDTree(descriptors[:,1:])
    
    #search for closest vector in descriptors to vector provided
    x = tree.query(vector)
    
    #reject any distance higher than 1.1
    if x[0] <= 1.1:
        #print the label associated with the found descriptor
        label = descriptors[x[1],0].astype(str)
        print("result for "+file+":")
        print("\t"+label+ " km/h\n")
        return label
    else:
        return "REJECTED"
    
#run main on script call
if __name__ == "__main__":
    for filename in listdir('roi'):
        if(filename.endswith('.png')):
            classifier("roi/",filename)
