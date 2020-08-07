import cv2
import numpy as np
from utils import plot_motion
import os



def get_flowmap(optflow_name):
    """
    通过光流算法获取云导风矢量图
    :param optflow_name: 光流算法名称
    :return:
    """
    data_root_path = "../datasets/data/CMWD/TRAIN_A"
    os.mkdir(os.path.join("../checkpoints",optflow_name))
    os.mkdir(os.path.join("../checkpoints", optflow_name, 'flowmap'))
    # os.mkdir(os.path.join("../checkpoints", optflow_name, 'prediction'))


    for i in range(len(os.listdir(data_root_path))):
        img = cv2.imread(os.path.join(data_root_path,str(i+1)+".png"))
        img_pre_3,img_next_3 = img[:,0:512,:],img[:,512:1024,:]
        img_previous = cv2.cvtColor(img_pre_3, cv2.COLOR_BGR2GRAY)
        img_next = cv2.cvtColor(img_next_3, cv2.COLOR_BGR2GRAY)

        flow = None
        if optflow_name == 'deepflow':
            df = cv2.optflow.createOptFlow_DeepFlow()
            flow = df.calc(img_previous, img_next, None)

        elif optflow_name == 'pcaflow':
            df = cv2.optflow.createOptFlow_PCAFlow()
            flow = df.calc(img_previous, img_next, None)

        else:
            raise ValueError("optflow_name mismatched!")

        # 通过第一帧图像计算外推后的第二帧图像并保存
        # y_coords, x_coords = np.mgrid[0:512, 0:512]
        # coords = np.float32(np.dstack([x_coords, y_coords]))
        # pixel_map = coords - flow
        # new_frame = cv2.remap(img_previous, pixel_map, None, cv2.INTER_CUBIC)
        # cv2.imwrite(os.path.join('../checkpoints',optflow_name,'prediction',str(i+1)+".png"), new_frame)

        # 对获得的光流进行后处理
        flow = cv2.resize(flow, (512, 512), cv2.INTER_CUBIC)
        flow = np.transpose(flow, (2, 0, 1))
        # 保存光流为numpy文件
        np.save(os.path.join("../checkpoints",optflow_name,'flowmap',str(i+1)+".npy"),flow)

        # 保存光流矢量图像
        # plot_motion.plot_motion_cloud(img_size=512,
        #                               motion_vector=new_flow * 5.,
        #                               savepath=os.path.join('../checkpoints',optflow_name,'flowmap',str(i+1)+'.png'),
        #                               bg=img_pre_3,
        #                               color='#DC143C',
        #                               plot_interval=8,
        #                               dpi=50,
        #                               filters=0)

        print("Already get flow and prediction:"+str(i+1)+"/715")




get_flowmap(optflow_name='pcaflow')
