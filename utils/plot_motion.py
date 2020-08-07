import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch



def plot_motion_cloud(motion_vector,savepath,img_size=512,bg=None,dpi=300,color="#DC143C",plot_interval=8,plot_size=32,limits=0):
    """
        将矢量场图进行可视化显示和保存
    :param motion_vector: (ndarray) 矢量场的位移坐标：(2, any_size, any_size)
    :param savepath: (str) 保存图像的路径
    :param img_size: (int) 图像的大小,云导风应用中默认为512
    :param bg: (ndarray) 背景图像:(img_size, img_size, 3)
    :param dpi: (int) 保存图像的dpi,默认为300
    :param color: 风矢箭头的颜色，默认为红色
    :param plot_interval:(int) 相隔多少个像素绘制矢量，大于1可以稀疏流场(建议卫星云图设置为8,默认设置为8)
    :param plot_size:(float) 保存图像的大小，单位为x100像素,默认为10,即保存图像的大小为3200*3200
    :param limits: (float)此参数用于稀疏较短的矢量场，如果矢量长度小于 limits，此处的矢量将会被忽略,默认为0(不进行限制)
    :return:
    """
    if np.shape(motion_vector)[0] != 2:
        raise ValueError("motion_vector size error!")
    if (bg is not None) and (np.shape(bg) != (img_size, img_size, 3)):
        raise ValueError('bg image size error!')

    # 自动设置光流矢量的大小
    vector_size = (2,int(img_size/plot_interval),int(img_size/plot_interval))
    new_motion_vector = np.ndarray(shape=vector_size,dtype=np.float)
    if np.shape(motion_vector) != vector_size:
        new_motion_vector[0] = cv2.resize(motion_vector[0],vector_size[1:])
        new_motion_vector[1] = cv2.resize(motion_vector[1],vector_size[1:])
    else: new_motion_vector = motion_vector

    new_flow = np.ndarray(shape=np.shape(new_motion_vector), dtype=np.float)
    for m in range(vector_size[2]):
        new_flow[:, m, :] = new_motion_vector[:, vector_size[2]-m-1, :]
    new_flow[0], new_flow[1] = -1 * new_flow[0], -1 * new_flow[1]


    # 筛选长度小于limits的矢量不进行显示
    if limits > 0:
        for i in range(int(img_size/plot_interval)):
            for k in range(int(img_size/plot_interval)):
                if np.sqrt(np.power(motion_vector[0][i][k],2)+np.power(motion_vector[1][i][k],2)) < limits:
                    new_flow[0][i][k] = 0
                    new_flow[1][i][k] = 0


    x0 = np.arange(plot_interval/2., img_size, plot_interval)
    x1 = np.arange(plot_interval/2, img_size, plot_interval)
    X, Y = np.meshgrid(x0, x1)
    X, Y = X.flatten(), Y.flatten()

    new_flow = np.reshape(new_flow,newshape=(2,-1))

    plt.figure(figsize=(plot_size,plot_size))
    if bg is not None:
        plt.imshow(bg, extent=[0, img_size, 0, img_size])

    plt.quiver(X,Y,-new_flow[0],new_flow[1],angles='xy', scale_units='xy', scale=1, color=color)
    plt.xlim([0,img_size])
    plt.ylim([0,img_size])
    plt.axis("equal")   # 横坐标和纵坐标一致
    plt.axis('off')     # 取消显示坐标轴
    plt.draw()
    plt.savefig(savepath,dpi=dpi)
    plt.close()




def plot_motion_field(motion_vector,savepath,img_size=512,bg=None,dpi=300,color="#DC143C",plot_interval=8,plot_size=32,limits=0):
    """
        将矢量场图进行可视化显示和保存
    :param motion_vector: (ndarray) 矢量场的位移坐标：(2, any_size, any_size)
    :param savepath: (str) 保存图像的路径
    :param img_size: (int) 图像的大小,云导风应用中默认为512
    :param bg: (ndarray) 背景图像:(img_size, img_size, 3)
    :param dpi: (int) 保存图像的dpi,默认为300
    :param color: 风矢箭头的颜色，默认为红色
    :param plot_interval:(int) 相隔多少个像素绘制矢量，大于1可以稀疏流场(建议卫星云图设置为8,默认设置为8)
    :param plot_size:(float) 保存图像的大小，单位为x100像素,默认为10,即保存图像的大小为3200*3200
    :param limits: (float)此参数用于稀疏较短的矢量场，如果矢量长度小于 limits，此处的矢量将会被忽略,默认为0(不进行限制)
    :return:
    """
    if np.shape(motion_vector)[0] != 2:
        raise ValueError("motion_vector size error!")
    if (bg is not None) and (np.shape(bg) != (img_size, img_size, 3)):
        raise ValueError('bg image size error!')

    # 自动设置光流矢量的大小
    vector_size = (2,int(img_size/plot_interval),int(img_size/plot_interval))
    new_motion_vector = np.ndarray(shape=vector_size,dtype=np.float)
    if np.shape(motion_vector) != vector_size:
        new_motion_vector[0] = cv2.resize(motion_vector[0],vector_size[1:])
        new_motion_vector[1] = cv2.resize(motion_vector[1],vector_size[1:])
    else: new_motion_vector = motion_vector

    new_flow = np.ndarray(shape=np.shape(new_motion_vector), dtype=np.float)
    for m in range(vector_size[2]):
        new_flow[:, m, :] = new_motion_vector[:, vector_size[2]-m-1, :]
    new_flow[0], new_flow[1] = -1 * new_flow[0], -1 * new_flow[1]


    # 筛选长度小于limits的矢量不进行显示
    if limits > 0:
        for i in range(int(img_size/plot_interval)):
            for k in range(int(img_size/plot_interval)):
                if np.sqrt(np.power(motion_vector[0][i][k],2)+np.power(motion_vector[1][i][k],2)) < limits:
                    new_flow[0][i][k] = 0
                    new_flow[1][i][k] = 0


    x0 = np.arange(plot_interval/2., img_size, plot_interval)
    x1 = np.arange(plot_interval/2, img_size, plot_interval)
    X, Y = np.meshgrid(x0, x1)
    plt.figure(figsize=(plot_size,plot_size))
    if bg is not None:
        plt.imshow(bg, extent=[0, img_size, 0, img_size])
    plt.streamplot(X, Y, -new_flow[0],new_flow[1], density=1.5, color=color,arrowsize=1.2,linewidth=1.5)
    plt.xlim([0,img_size])
    plt.ylim([0,img_size])
    plt.axis("equal")   # 横坐标和纵坐标一致
    plt.axis('off')     # 取消显示坐标轴
    plt.draw()
    plt.savefig(savepath,dpi=dpi)
    plt.close()



def viz_flow(motion_flow):
    # motion_flow with size (2,height,width)
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    motion_flow = np.swapaxes(motion_flow,0,2)
    motion_flow = np.swapaxes(motion_flow,0,1)
    h, w = motion_flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(motion_flow[...,0], motion_flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr



def vis_flow_tensor(motion_flows):
    # input motion_flows with size(batchsize,2,height,width)
    motion_flows = torch.split(motion_flows,split_size_or_sections=1,dim=0)
    bgrs = []
    for i in range(len(motion_flows)):
        motion_flow_tmp = motion_flows[i].squeeze(0)
        motion_flow_tmp = motion_flow_tmp.cpu().data.numpy()
        bgr = viz_flow(motion_flow_tmp)
        bgr = torch.from_numpy(bgr)
        bgrs.append(bgr)

    bgrs = torch.stack(bgrs,dim=0)
    bgrs = bgrs.permute(0,3,1,2)
    return ((bgrs /255.)-0.5)*2.
