# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:50:14 2022

@author: ASUS
"""
#import imageio as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as transform
from PIL import Image,ImageOps

im = Image.open("speed-sign-test-images/40-0001x1.png")

def main():
    plt.imshow(im)
    plt.show()
    greyIm = im.convert('L')
    resizedIm = ResizeImage(greyIm)
    Vectorisation(resizedIm)

#change the image to 64x64 and greyscale
def ResizeImage(im):
    #change the original image to greyscale
    newIm = ImageOps.grayscale(im)
    #shrink the image to 64x64 pixels
    newIm = transform.resize(np.asarray(im),(64,64))
    #print the new image size
    print("new image size in pixels:   ",newIm.shape)
    #display the image
    plt.imshow(newIm,cmap='gray')
    plt.show()
    return newIm
    
#transform the numpy array into a 4096-dimensional vector
def Vectorisation(im):
    #scale the elements of the image array to fit between 0 and 255
    im = (im*255)
    plt.imshow(im,cmap='gray')
    plt.show()
    
#run main on script call
if __name__ == "__main__":
    main()