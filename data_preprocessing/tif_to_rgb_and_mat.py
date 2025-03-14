# This code will separate the RGB image and the corresponding spectral images from the HSI cube stored as multipage tif format
import shutil

import numpy

from utils import save_matv73
from tiff_utility import read_stiff
from matplotlib import pyplot as plt
from PIL import Image
import glob
import os
import numpy as np
from scipy.io import savemat
from tifffile import imread, imwrite
from spectral import calc_stats, noise_from_diffs, imshow, mnf
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dataset_path = "D://PhD_Projects//HSI_Project//vein-visualization-master//oral_dental_reconstruction//MNF_based_ODSI_Reconsturction//Selected_and_Resized_Images"  # This dataset_path is the folder where the resized HSIs of specified bandwidth are stored

rgbfiles_path = dataset_path + "//RGB"
matfiles_path = dataset_path + "//MAT"
mnf_as_image = dataset_path + "//mnfs"

files = os.listdir(dataset_path)

if os.path.exists(rgbfiles_path):
    shutil.rmtree(rgbfiles_path)
os.makedirs(rgbfiles_path)

if os.path.exists(matfiles_path):
    shutil.rmtree(matfiles_path)
os.makedirs(matfiles_path)

if os.path.exists(mnf_as_image):
    shutil.rmtree(mnf_as_image)
os.makedirs(mnf_as_image)


def apply_mnf(m, num):
    m_new = m.copy()
    signal = calc_stats(m_new)
    noise = noise_from_diffs(m_new)
    mnfr = mnf(signal, noise)
    reduced = mnfr.reduce(m_new, num=num)

    return reduced

for file in files:
    print(file)
    spim, array_of_wavelength, rgb, metadata = read_stiff(dataset_path+"//"+file)  # Now, the spim variable has 43 bands. Because in the range of 620 to 750 nm range, 43 bands were found.

    # saving rgb image as jpg from tif file
    filename = file.split('.tif')[0]  # The name of the file of the RGB image is considered
    img = Image.fromarray(rgb, 'RGB')  # The rgb image(numpy array) is converted to PIL image object using the fromarray function
    img.save(rgbfiles_path+ "//" +filename+".jpg")  # The rgb image is saved in specific folder path

    mnf_result = apply_mnf(spim, 3)
    Image.fromarray(mnf_result, 'RGB').save(mnf_as_image+"//"+filename+".jpg")
    # plt.imsave(file+".jpg", mnf_result/255)
    # saving spectral image as mat file from tif file
    tmp = {
        'rad': mnf_result   # The spectral images are saved as a dictionary named tmp. 'rad' only the key name. Each spim will be 512 x 512 x 3 for the images in folder band_620_750
    }

    save_matv73(matfiles_path+ "//" +filename+".mat",'rad', mnf_result)
    # savemat(matfiles_path+ "//" +filename+".mat", tmp)

    # After the code is run, the RGB image will be saved in img variable and the spectral images will be stored in the spim variable.
