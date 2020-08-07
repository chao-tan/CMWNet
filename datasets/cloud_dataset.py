import os.path
from datasets.cdataset import cdataset
import numpy as np
import torch
from torchvision import transforms
import cv2
import random


def normalization(data):
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]

    transform = transforms.Normalize(mean,std,inplace=False)
    return transform(data)



class cloudDataset(cdataset):
    def __init__(self, config):
        cdataset.__init__(self, config)
        self.input_paths = os.path.join(config['dataroot'], str.upper(config['status']+"_A"))
        self.flow_paths = os.path.join(config['dataroot'],str.upper(config['status']+"_B"))
        self.dataset_len = len(os.listdir(self.input_paths))
        self.config = config

    def __getitem__(self, index):
        img_ = cv2.imread(os.path.join(self.input_paths,str(index+1)+".png")).astype(np.float)
        img_x = img_[:,0:512,:]
        img_y = img_[:,512:1024,:]
        flow_ = np.load(os.path.join(self.flow_paths,str(index+1)+".npy")).astype(np.float)
        flow_x = flow_[0,:,:]
        flow_y = flow_[1,:,:]

        if self.config['status'] == 'train':
            crop_size = int(self.config['img_size']*self.config['crop_prob'])
            img_x = cv2.resize(img_x,(crop_size,crop_size))
            img_y = cv2.resize(img_y,(crop_size,crop_size))
            flow_x = cv2.resize(flow_x,(crop_size,crop_size))
            flow_y = cv2.resize(flow_y,(crop_size,crop_size))

            ranp = random.sample(range(int((crop_size-int(self.config['img_size']))*0.5)),2)
            img_x = img_x[ranp[0]:ranp[0]+int(self.config['img_size']),ranp[1]:ranp[1]+int(self.config['img_size'])]
            img_y = img_y[ranp[0]:ranp[0]+int(self.config['img_size']),ranp[1]:ranp[1]+int(self.config['img_size'])]
            flow_x = flow_x[ranp[0]:ranp[0]+int(self.config['img_size']),ranp[1]:ranp[1]+int(self.config['img_size'])]
            flow_y = flow_y[ranp[0]:ranp[0]+int(self.config['img_size']),ranp[1]:ranp[1]+int(self.config['img_size'])]


        # 将numpy图像转换为tensor
        img_x = torch.from_numpy(img_x).float()/255.
        img_y = torch.from_numpy(img_y).float()/255.
        img_x = img_x.permute(2,0,1)
        img_y = img_y.permute(2,0,1)
        flow_x = torch.from_numpy(flow_x).float()
        flow_y = torch.from_numpy(flow_y).float()
        flow = torch.cat([flow_x.unsqueeze(0),flow_y.unsqueeze(0)],dim=0)
        # 标准化数据
        img_x = normalization(img_x)
        img_y = normalization(img_y)

        return {'IMAGE_X': img_x,
                "IMAGE_Y": img_y,
                'FLOW':flow,
                'PATH': index + 1,
                }


    def __len__(self):
        return self.dataset_len


# from utils import startup
# from datasets import cloud_dataset
# import cv2
# import numpy as np
# from utils import plot_motion
# config = startup.SetupConfigs(config_path='configs/_cloud.yaml').setup()
# dataset = cloud_dataset.cloudDataset(config=config)
# print(dataset.__len__())
# datapoint = dataset.__getitem__(index=150)
# image_x = datapoint["IMAGE_X"]
# image_y = datapoint['IMAGE_Y']
# flow = datapoint['FLOW']
# print(datapoint['PATH'])
#
# image_x = image_x.permute(1,2,0)
# image_x = image_x.data.cpu().numpy()
# image_x = (image_x + 1)*127.
#
# image_y = image_y.permute(1,2,0)
# image_y = image_y.data.cpu().numpy()
# image_y = (image_y + 1)*127.
#
# flow = flow.data.cpu().numpy()
#
# cv2.imwrite("D://image_x.png",image_x)
# cv2.imwrite("D://image_y.png",image_y)
# plot_motion.plot_motion_cloud(img_size=512,
#                               motion_vector=flow*5.,
#                               savepath='D://flow.png',
#                               bg=image_x.astype(np.int),
#                               plot_interval=8,
#                               dpi=20)
#
#
# cv2.imwrite("D://flow_black.png",flow[0]*10.)
# cv2.imwrite("D://vis.png",plot_motion.viz_flow(flow))