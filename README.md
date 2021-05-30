# Learning to Efficient Cloud Motion Wind using Deep Learning Approach


This repository contains the source code, dataset and pretrained model for CMWNet, provided by [Chao Tan](https://chao-tan.gitee.io).

The paper is avaliable for download [here](https://arxiv.org/abs/2010.01283). 
Click [here](https://chao-tan.gitee.io/projects/cmw-net/project-page.html) for more details.


***
## Dataset & Pretrained Model
CMWD (Cloud Motion Wind Dataset) is the first cloud motion wind dataset for deep learning research.
It contains 6388 adjacent grayscale image pairs for training and another 715 images pairs for testing.
Our CMWD dataset is available for download at [TianYiCloud(2.2GB)](https://cloud.189.cn/t/NRbARji2A3Qv) or [BaiduCloud(2.2GB)](https://pan.baidu.com/s/1B-W-OBHuKxqbfJktLqAnJQ) (extraction code: np6o).      
You can get the CMWD dataset at any time but only for scientific research. 
At the same time, please cite our work when you use the CMWD dataset.

The pretrained model of our CMWNet on CMWD dataset can be download at [TianYiCloud](https://cloud.189.cn/t/zyI3iyNZVV7n) or [BaiduCloud](https://pan.baidu.com/s/18kY_Li0myN5P1xgklbKufw) (extraction code: wqk0).

  
## Prerequisites
* Python 3.7
* PyTorch >= 1.4.0
* opencv 0.4
* PyQt 4
* numpy
* visdom


## Training
1. Please download and unzip TCLD dataset and place it in ```datasets/data``` folder.
2. Generating labels for CMWD dataset.
    - Since our CMWD dataset does not explicitly give a cloud motion wind label for each image pair, 
    you can use existing methods (such as any optical flow algorithm) to generate pixel-wise motion vectors between two frames and use it as the traing label.
    Please save the motion vectors as an numpy array of size (2* image_length * image_width) under the ```TRAIN_B``` and ```TEST_B``` folders respectively. 
    The name of each generated label is the same as the input satellite image.
3. Run ```python -m visdom.server"``` to activate visdom server.
4. Run ```python run.py``` to start training from scratch.
5. You can easily monitor training process at any time by visiting ```http://localhost:8097``` in your browser.


## Testing
1. For TCLD dataset, please download and unzip pretrained model and place it in ```checkpoints``` folder.
2. You need to modify the ```configs/FlowNetS.yaml``` file and change the status option from train to test.
3. Run ```python run.py``` to start testing.
4. The results of the testing will be saved in the ```checkpoint/FlowNetS/testing"``` directory.

## Citation
@inproceedings{      
&nbsp;&nbsp;&nbsp;&nbsp;  title={Generating the Cloud Motion Wind Field from Satellite Cloud Imagery Using Deep Learning Approach},         
&nbsp;&nbsp;&nbsp;&nbsp;  author={Tan, Chao},         
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle={IGARSS},        
&nbsp;&nbsp;&nbsp;&nbsp;  year={2021},        
&nbsp;&nbsp;&nbsp;&nbsp;  note={to appear},       
}



