### Project Objective ###

Reliable visual discrimination of clinical tissue types is a challenging problem because traditional RGB imaging captures information restricted to the human eye's capabilities. Hyperspectral imaging(HSI) can capture abundant visual information using many narrow bands, facilitating tissue differentiation. However, till now, hyperspectral imaging has been successful mainly in massive industrial applications due to the high cost of hyperspectral cameras. In this project, we propose a deep learning-based approach to reconstruct hyperspectral images from corresponding RGB images. We employ the Oral and Dental Spectral Image Database (ODSI-DB), consisting of 316 oral and dental reflectance spectral images, to achieve this. We apply the minimum noise fraction (MNF) dimensionality reduction approach on the ODSI datacubes and consider the top three MNF-generated images as those exhibiting the most informative features. Instead of training the proposed deep learning model with all the spectral images, we utilize the MNF-generated images as groundtruth and reconstruct three images. Evaluation shows that our method can reconstruct three HSIs close to the groundtruths and preserve informative features with high-class separability. 

### Dataset Description ###
We utilize the publicly available Oral and Dental Spectral Imaging Database(ODSI-DB). The dataset can be downloaded form [here](https://cs.uef.fi/pub/color/spectra/ODSI-DB/). 

A [preview tool](https://cs.uef.fi/pub/color/spectra/ODSI-DB/preview-tool-v2/) is provided by the dataset creators to visualize different tissue regions. 

The ODSI dataset contains 316 oral and dental hyperspectral images. The Specim IQ line scan camera captured 171 images with 204 spectral channels in the 400 - 1000nm range with a spatial resolution of 512x512. The remainder of the images were taken with another camera in the 450 - 950nm range with 51 spectral channels and 1392x1040 spatial resolution. 

### Data Pre-processing ###

The ODSI images are first resized. Run the **data_preprocessing\Resize_ODSI_files.py** file. 

![MNF generation figure](https://github.com/hafsa390/ODSI_Reconstruction/blob/main/images/mnf_figure_final.JPG)

The top three images are considered as ground truth of three channels of the RGB images for reconstructing HSIs.

### Network Architecture ###

The proposed model is inspired by [deep residual U-Net](https://arxiv.org/pdf/1711.10684) and [MD-UNet](https://www.sciencedirect.com/science/article/abs/pii/S1476927121000773). The deep residual U-Net uses residual units consisting of several combinations of convolutional layer followed by batch normalization (BN), and rectified linear unit (ReLU) activation.

We used a 7-layer deep architecture consisting of an encoder path, decoder path, bridge, and skip connection. The encoder path has three residual units. The first residual unit consists of a convolutional layer, BN layer, ReLU, and another convolutional layer. The other two residual units are similar to the first one, except they have one BN layer followed by ReLU at the beginning. The residual unit of the bridge block is the same as the encoder path's second or third residual unit. Likewise, the residual units of the decoder path consist of two subsequent combinations of the BN layer, convolutional layer, and ReLU. An upsampling layer is applied before each residual unit of the decoder path. A dropout value of 0.20 is added to the output of each of the residual units. 

The strided convolution is used in each residual unit of encoder path to downsample feature maps. To incorporate multiple input streams, the original input image is resized to match the size of the feature maps in the second and third residual units in the encoder path, convoluted with a $1 \times 1$ convolutional kernel, and concatenated with the output of the previous layer. A skip connection is used to concatenate the output of each unit on the encoder path with the corresponding layer of the decoder path.

![The proposed UNet figure](https://github.com/hafsa390/ODSI_Reconstruction/blob/main/images/unet_final.JPG)

### Model Training ###

The proposed network is trained using 64x64 overlapping patches having a stride 32. The batch size is 32. The Adam optimizer is used with beta1 = 0.9 and beta2 = 0.999$. The initial learning rate is 0.0001 with the polynomial function as the decay policy. The model is trained for 200 epochs. The average runtime of each epoch is 328.330872536s.

### Results ###
Proposed Model & 2.717 $\pm$ 1.7750 & 8.684 $\pm$ 0.7709 \\ \hline 
Vanilla U-Net & 3.095 $\pm$ 1.8792 & 8.094 $\pm$ 0.9352 \\ 

|Model          |MRAE                |PSNR               |
|---------------|--------------------|-------------------|
|Proposed Model | 2.717 $\pm$ 1.7750 | 8.684 $\pm$ 0.7709|
|Vanilla UNet   | 3.095 $\pm$ 1.8792 | 8.094 $\pm$ 0.9352|
