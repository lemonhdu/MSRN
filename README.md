# MSRN
This is the pytorch implemented code for MSRN. The network is tested on UNLV-Dive Dataset.

# Usage
## Requirements
1. Pytorch >= 1.12.1
2. Scipy >= 1.4.1

## Dataset Preparation
### prepare dataself yourself
1. Download the UNLV-Dive Dataset from [UNLV-Dive](http://rtis.oit.unlv.edu/datasets.html).
2. Try to use [Resnet3d](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet) to extract the features of videos. The features of the video in the paper are extracted frame by frame. The frame images are centrally cropped and 224×224 images are fed into ResNet3d. You should set the frequency parameter in main.py as 1, and set the parameters to adjust the cropping size in extract_features.py. We provide this two modified files in ./tools directory, while other files should be down from [Resnet3d](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet).
3. The extracted feature files should be named as xxx.npy, such as 001.npy.

### Using our preprocessing dataset
1. You can also download our preprocessing files from [google cloudDisk](https://drive.google.com/drive/folders/1z0U59MkXV-alxveIsYQk6zmRBUMQo0zv?usp=sharing).

## Train MS-TCN
1. Run the train_mstcn.py to train the MS-TCN network, you can stop the iteration if you feel satisfied about the action segmentation results. Some parameters in terms of file directories may need to be modified in params.py. We also provide a pretrained MS-TCN model for UNLV-Dive Dataset in ./ckpts directory, which can achieve a nearly 98% accuracy. 

### Train the score prediction network
1. Run train_minus_fc.py. The program can run without any preset parameters. Some parameters about training can be modified in config.py.

# Acknowledgement
We would thank Xiang et al. for their work [S3D](https://github.com/YeTianJHU/diving-score), which annotated the action transition time points for UNLV-Dive Dataset.

# Contact
We would provide the contact E-mail later since the submission of paper requires to be anonymous.


