import os
import shutil
import matplotlib.pyplot as plt
from tiff_utility import band_selection_resize, read_stiff
from PIL import Image
import numpy as np
from numpy import asarray

width = height = 512

ODSI_dataset_path = "D://PhD_Projects//HSI_Project//Oral_HSI//ODSI_Dataset"
destination_folder = "Selected_and_Resized_Images"
destination_path = ODSI_dataset_path + "//" + destination_folder


if not os.path.exists(destination_folder):        # Creating folder to save the resized images of selected bandwidth
    os.makedirs(destination_folder)


for file in os.listdir(ODSI_dataset_path):
    if file.endswith('tif') and file.__contains__('similarity') is False:
        spim, bands, rgb, metadata = read_stiff(ODSI_dataset_path + "//" + file)

        if len(bands > 200):
            band_selection_resize(ODSI_dataset_path + "//" + file, destination_folder + "//" + file, bands, width, height, 300, 1500)


