# YOLOv1_CCPD

# 环境配置：

python == 3.8

keras == 2.4.3

tensorflow == 2.4.1

opencv-python == 4.5.3.56


# 文件介绍： 

先运行main.py搭建模型进行训练，再运行yolo_predict.py调用训练权重进行检测。

1、crop_ccpd文件夹：存储车牌号3316张，目标检测数据集。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

2、demo文件夹：测试集的目标检测结果，效果不是很理想，主要是权重训练效果不佳。

3、Logs文件夹：存储训练过程中的权重文件。

4、raw_weights.hdf5：初始训权重文件，成功加载后，将网络的特征提取部分冰冻起来，此部分不再训练。

5、tiny_yolov1_model.py：tiny_yolov1模型，其特征提取backbone比yolov1简单很多，但在VOC2007数据集下仍有一定效果。

6、yolo_loss.py：yolov1损失函数，对无目标物体置信度加权系数0.5，有目标物体置信度加权系数1，类别加权系数1，物体x、y、w、h坐标位置加权系数5。

7、train.py文件：导入初始权重raw_weights.hdf5，冰冻特征提取部分前31层，优化器adam = Adam(lr=1e-4, amsgrad=True)效果最佳。

8、yolo_predict.py：检测测试集的目标检测效果。


# 训练效果：

优化器：adam = Adam(lr=1e-4, amsgrad=True)，训练150个epoch，最佳val loss在16.25，后续达到瓶颈无法降低。

YOLOv1最大的问题就在于，模型太难以训练，val loss难以降下去。
