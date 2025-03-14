from __future__ import division
import torch
import torch.nn as nn
import os
import time
import numpy as np
from imageio import imread
import PIL.Image
import shutil

import save_mat_to_jpg
from resblock import resblock,conv_bn_relu_res_block
from utils import save_matv73,reconstruction,load_mat,mrae,rmse

model_folder_name = "Three_Channel_rrmse_updated_not_divide_4095"
model_path = os.path.join(r"D:\PhD_Projects\HSI_Project\vein-visualization-master\oral_dental_reconstruction", model_folder_name)+"//HS_veins_120.pkl"


img_path = 'D://PhD_Projects//HSI_Project//vein-visualization-master//Visualization_and_Testing//test//rgb'
result_path = '../dataset/veins_t34bands/test_data/inference'
gt_path = 'D://PhD_Projects//HSI_Project//vein-visualization-master//Visualization_and_Testing//test//mat'
var_name = 'rad'

save_point = torch.load(model_path)
model_param = save_point['state_dict']

model = resblock(conv_bn_relu_res_block,10,3,3)
# print(model)
model.load_state_dict(model_param)

model = model.cuda()
model.eval()


for img_name in sorted(os.listdir(img_path)):

    # if img_name != 'test_hand_image.jpg':
    #     continue

    img_path_name = os.path.join(img_path, img_name)
    rgb = imread(img_path_name)   
    rgb = rgb/255
    rgb = np.expand_dims(np.transpose(rgb, [2,1,0]), axis=0).copy()

    img_res1 = reconstruction(rgb,model)
    img_res2 = np.flip(reconstruction(np.flip(rgb, 2).copy(),model),1)
    img_res3 = (img_res1+img_res2)/2

    # mat_name = img_name[:-4] + '.mat'
    mat_name = img_name + '.mat'
    mat_dir= os.path.join(result_path, mat_name)

    save_matv73(mat_dir, var_name,img_res3)

    # gt_name =  img_name[12:-4] + '.mat'
    # img_name = img_name.split('.jpg')[0]
    # gt_name =  img_name + '.mat'
    # gt_dir= os.path.join(gt_path, gt_name)
    #
    # # print(gt_dir)
    # gt = load_mat(gt_dir,var_name)
    # if len(gt['rad'].shape) > 2:
    #     mrae_error =  mrae(img_res3, gt['rad'][:,:,1])
    #     rrmse_error = rmse(img_res3, gt['rad'][:, :, 1])
    # else:
    #     mrae_error = mrae(img_res3, gt['rad'])
    #     rrmse_error = rmse(img_res3, gt['rad'])
    # # rrmse_error, scikit_rmse_error, MSE, scikit_mae, scikit_SSIM = rmse(img_res3, gt['rad'][:,:,1])
    #
    # print("[%s] MRAE=%0.9f RRMSE=%0.9f" %(img_name,mrae_error,rrmse_error))

#     print("[%s]:", img_name, "scikit learn RMSE and MSE", scikit_rmse_error, MSE)
#     print("[%s]:", img_name, "scikit learn MAE", scikit_mae)
#     print("[%s]:", img_name, "scikit image SSIM", scikit_SSIM)


# save_mat_to_jpg.run("temp", 'temp')