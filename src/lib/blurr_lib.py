import cv2
import os
import pandas as pd
import numpy as np

from lib import general_lib, blur_wavelet_lib

import cpbd

from PIL import Image


def motion_classifier(img):
    per, blur_extent, classification = blur_wavelet_lib.blur_detect(img, 0.001)
    return round(10000*per)
    

def laplacian_classifier(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurr = round(cv2.Laplacian(gray, cv2.CV_64F).var())
    return blurr
    
    
def cpbd_classifier(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurr = round(1000 * cpbd.compute(gray))
    return blurr    


def sharp_classifier(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    array = np.asarray(gray, dtype=np.int32)
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = round(10 * np.average(gnorm))
    return sharpness
    

def grid_picture(image, grid_size=[3,3]):
    grid_rows = grid_size[0]
    grid_cols = grid_size[1]
    
    wide_base = image.shape[0]
    high_base = image.shape[1]
    
    wide_sub = round(high_base/grid_rows)
    high_sub = round(wide_base/grid_cols)
    
    image_list_names = []
    image_list_grid = []
    
    for row in range(grid_rows):
    	for col in range(grid_cols):   	    
    	    image_sub = image[row*high_sub:(row+1)*high_sub, col*wide_sub:(col+1)*wide_sub]
    	    image_list_grid.append(image_sub)
	
    return image_list_grid


def blurr_classifier_grid(image, grid_size=[3,3], cpbd=True, motion=True):
    '''
    Returns the minimum blurr on a given image divided by grid.
    This is meant to solve the problem on detecting blurr surrounding a focused object.
    '''

    image_list_grid = grid_picture(image, grid_size)
    laplacian_list = []
    cpbd_list = []
    sharp_list = []
    
    for image_sub in image_list_grid:
    	laplacian_sub = laplacian_classifier(image_sub)
    	laplacian_list.append(laplacian_sub)
    	
    	
    max_laplacian = np.max(laplacian_list)
    mean_laplacian = round(np.mean(laplacian_list))
    
    cpbd_value = 0
    if cpbd:
    	cpbd_value = cpbd_classifier(image)

    motion_value = 0
    if motion:
    	motion_value = motion_classifier(image)    
    
    return max_laplacian, mean_laplacian, cpbd_value, motion_value


def blurr_classifier_folder(path, grid_size=[3,3]):
    image_list = os.listdir(path)
    image_list = general_lib.filter_images(image_list)

    max_laplacian_list = []
    mean_laplacian_list = []
    cpbd_list = []
    motion_list = []
        
    for image_name in image_list:
        image = cv2.imread(path + '/' + image_name)
        max_laplacian, mean_laplacian, cpbd_value, motion_value = blurr_classifier_grid(image, grid_size)
        max_laplacian_list.append(max_laplacian)
        mean_laplacian_list.append(mean_laplacian)
        cpbd_list.append(cpbd_value)     
        motion_list.append(motion_value)   
        
    blurr_df = pd.DataFrame({'image_name': image_list, 'max_laplacian': max_laplacian_list, 'mean_laplacian': mean_laplacian_list, 'cpbd': cpbd_list, 'motion': motion_list})
    path_save = 'output/results'
    general_lib.create_folder(path_save)
    blurr_df.to_csv(path_save+'/pictures_blur.csv')
    
    return blurr_df
    


def blurr_sort(path, threshold=[150,100,200]):
    blurr_df = pd.read_csv('output/results/pictures_blur.csv', index_col=0)

    for index in range(blurr_df.shape[0]):
        blurry_laplacian = blurr_df.max_laplacian[index]
        blurry_cpbd = blurr_df.cpbd[index]
        blurry_motion = blurr_df.motion[index]
        is_motion = (blurry_motion > threshold[2]) and (blurry_laplacian < 1000)

        if blurry_laplacian < threshold[0] or blurry_cpbd < threshold[1] or is_motion:
            general_lib.move_file(blurr_df.image_name[index], path, path + '/blurry')
