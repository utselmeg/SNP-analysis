
import os
import numpy as np
import random
from utils import selection
from logger_config import logger

IMG_DIR = "/gpfs/gibbs/pi/gerstein/jrt62/imaging_project/expression-prediction/Thyroid/Thyroid-no-bounding-features-remove-bg"

def is_valid_patch(arr):
    # Check if a patch is valid
    return arr.shape[0] == 64 and arr.shape[1] == 64 and arr.shape[2] == 128 and np.isnan(arr[:,:,1]).sum() < 1000

def augment_patch(arr, y, sum_x, sum_y, is_train_data):
    # If the patch is valid, augment it and append it to sum_x and sum_y
    if is_valid_patch(arr):
        arr1 = np.nan_to_num(arr)
        sum_x.append(arr1)
        sum_y.append(y)
        if is_train_data: 
            num = random.randint(1,7)
            sum_x.append(selection(arr1, num))
            sum_y.append(y)

def process_slide(i, y_label, sum_x, sum_y, is_train_data):
    # Process a whole slide
    a = np.load(i + "_features.npy") # Load the compressed slide
    b = np.swapaxes(a, 0, 2) # Swap axes
    unit_list = []
    
    for j in range(128):
        c = b[:,:,j][~np.isnan(b[:,:,j]).all(axis=1)] # Remove NaN rows
        d = (c.T[~np.isnan(c.T).all(axis=1)]).T # Remove NaN columns
        unit_list.append(d)
        
    e = np.array(unit_list)
    f = np.swapaxes(e, 0, 2)
    
    f_len = f.shape[0]
    f_width = f.shape[1]
    f_len_int = f_len // 32
    f_width_int = f_width // 32
    
    for k in range(f_len_int - 2):
        for p in range(f_width_int - 2):
            patch100 = f[k * 32:(k * 32 + 64), p * 32:(p * 32 + 64),:]
            augment_patch(patch100, y_label, sum_x, sum_y, is_train_data)
            
        patch100 = f[k * 32:(k * 32 + 64), -65:-1,:]
        augment_patch(patch100, y_label, sum_x, sum_y, is_train_data)

    for p in range(f_width_int):
        patch100 = f[-65:-1,p * 32:(p * 32 + 64),:]
        augment_patch(patch100, y_label, sum_x, sum_y, is_train_data)
    patch100 = f[-65:-1,-65:-1,:]
    augment_patch(patch100, y_label, sum_x, sum_y, is_train_data)

def process_all_slides(data, labels, is_train_data):
    sum_x = []
    sum_y = []

    os.chdir(IMG_DIR)
    if is_train_data:
        logger.info("Processing all slides: training")
    else:
        logger.info("Processing all slides: testing")

    for index, _ in enumerate(data.iterrows()):
        # if is_train_data:
        #     print(index)
        # else:
        #     print(_)
        y_label = labels[index] # access label using integer index
        image_file = data.iloc[index]['image_file']
        process_slide(image_file, y_label, sum_x, sum_y, is_train_data)

    logger.info("Finished processing.")
    return sum_x, sum_y

def stack_data(sum_x, sum_y):
    x = np.stack(sum_x)
    y = np.array(sum_y)
    return x, y
