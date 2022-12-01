# MSRN
This is the pytorch implemented code for MSRN. This network is tested in UNLV-Dive Dataset.

# Usage
## Requirements
1. Pytorch >= 1.12.1
2. Scipy >= 1.4.1

## Dataset Preparation
### prepare dataself yourself
1. Download the UNLV-Dive Dataset from [UNLV-Dive](http://rtis.oit.unlv.edu/datasets.html).
2. Try to use [Resnet3d](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet) to extract the features of videos. The features of the video in the paper are extracted frame by frame. The frame images are centrally cropped and 224×224 images are fed into ResNet3d. You should set the frequency parameter in main.py as 1, and set the parameters to adjust the cropping size in extract_features.py. We provide this two files in ./tools directory, while other files should be down from [Resnet3d](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet).
3. The output file name should be XXX.npy, such as 001.npy.

### Prepare dataself yourself
1. You can also download the our preprocessing files from 

## Train MS-TCN
1. Run the train_mstcn.py to train the MS-TCN network, you can stop the iteration if you feel satisfied about the action segmentation results. Some file directories may need modify in params.py. We also provide a retrained MS-TCN for UNLV-Dive Dataset in ./ckpts directory, which can achieve a nearly 98% accuracy.

### Train the score prediction network
1. Run train_minus_fc.py, try to modify the pretrained MS-TCN model.

# Acknowledgement
We would thank Xiang et al. for their work of annotating the action transition time points in UNLV-Dive Dataset.

# Contact
If you have any questions about our work, please contact .


