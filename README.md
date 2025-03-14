### Project Objective ###

Reliable visual discrimination of clinical tissue types is a challenging problem because traditional RGB imaging captures information restricted to the human eye's capabilities. Hyperspectral imaging(HSI) can capture abundant visual information using many narrow bands, facilitating tissue differentiation. However, till now, hyperspectral imaging has been successful mainly in massive industrial applications due to the high cost of hyperspectral cameras. In this project, we propose a deep learning-based approach to reconstruct hyperspectral images from corresponding RGB images. We employ the Oral and Dental Spectral Image Database (ODSI-DB), consisting of 316 oral and dental reflectance spectral images, to achieve this. We apply the minimum noise fraction (MNF) dimensionality reduction approach on the ODSI datacubes and consider the top three MNF-generated images as those exhibiting the most informative features. Instead of training the proposed deep learning model with all the spectral images, we utilize the MNF-generated images as groundtruth and reconstruct three images. Evaluation shows that our method can reconstruct three HSIs close to the groundtruths and preserve informative features with high-class separability. 

### Dataset Description ###
We utilize the publicly available Oral and Dental Spectral Imaging Database(ODSI-DB). The dataset can be downloaded form [here](https://cs.uef.fi/pub/color/spectra/ODSI-DB/). 

A [preview tool](https://cs.uef.fi/pub/color/spectra/ODSI-DB/preview-tool-v2/) is provided by the dataset creators to visualize different tissue regions. 

The ODSI dataset contains 316 oral and dental hyperspectral images. The Specim IQ line scan camera captured 171 images with 204 spectral channels in the 400 - 1000nm range with a spatial resolution of 512x512. The remainder of the images were taken with another camera in the 450 - 950nm range with 51 spectral channels and 1392x1040 spatial resolution. 

### Data Pre-processing ###

The ODSI images are first resized. Run the **data_preprocessing\Resize_ODSI_files.py** file. 
