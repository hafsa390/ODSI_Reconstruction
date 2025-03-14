# Applying Minimum Noise Fraction(MNF) algorithm on the ODSI images and saving in particular subfolders

import os
import numpy as np
from sklearn import decomposition
from PIL import Image
import matplotlib.pyplot as plt
from spectral import calc_stats, noise_from_diffs, imshow, mnf
from HSI2RGB import HSI2RGB
from tiff_utility import read_stiff

ODSI_data_path = "D://PhD_Projects//HSI_Project//Oral_HSI//ODSI_Dataset//Full_Band_Specim_Camera"


def apply_mnf(m, num):
    m_new = m.copy()
    signal = calc_stats(m_new)
    noise = noise_from_diffs(m_new)
    mnfr = mnf(signal, noise)
    reduced = mnfr.reduce(m_new, num=num)
    return reduced


def separate_bands(M, title):
    imshow(M[:, :, 0], stretch=0.0, title=title + " BAND 1")
    plt.savefig(os.path.join(fullpath, 'mnf_band1.png'))
    imshow(M[:, :, 1], stretch=0.0, title=title + " BAND 2")
    plt.savefig(os.path.join(fullpath, 'mnf_band2.png'))
    imshow(M[:, :, 2], stretch=0.0, title=title + " BAND 3")
    plt.savefig(os.path.join(fullpath, 'mnf_band3.png'))
    return

for file in os.listdir(ODSI_data_path):
    if file.endswith('tif'):
        spim, bands, rgb, metadat = read_stiff(ODSI_data_path + "//" + file)

        print(type(spim))
        print(spim.shape)
        print(rgb.shape)



        Main_Directory_path = "D://PhD_Projects//HSI_Project//vein-visualization-master//oral_dental_reconstruction//Teemp"
        fullpath = Main_Directory_path + "//" + file + "//" + "0_1_2"



        mnf_result = apply_mnf(spim, 3)
        print(mnf_result.shape)
        imshow(mnf_result, stretch=0.0, bands=(0, 1, 2))
        plt.savefig(os.path.join(fullpath, 'mnf_result.png'))
        print(file)
        separate_bands(mnf_result, title="MNF")
#mnf_result_conv = mnf_result.astype(np.uint8) # converting the float to int values

# Numpy_arr_to_PIL_Image = Image.fromarray(mnf_result_conv)
# Numpy_arr_to_PIL_Image.show()

#separate_bands(Numpy_arr_to_PIL_Image, "Reordered MNF")
# print(rgb[0])
# print("Total number of bands:", bands.shape)
# print("Printing " + ODSI_data_path + "'s properties")
# print("Printing the type of spim", type(spim))
# print("The shape of spim", spim.shape)

# print(spim[0].shape)
# print(spim[0])
# img = spim[0]
# Numpy_arr_to_PIL_Image = Image.fromarray(img)
# Numpy_arr_to_PIL_Image.show()
# print(spim[0][0][0])
        # # print(spim[0].shape)
        # # print(spim[0][0].shape)
        # # print(spim[0][0][0].shape)
        # #print("Printing the entire spectral image", spim)

#print(Numpy_arr_to_PIL_Image.shape)