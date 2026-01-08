#coding:utf-8
from ultralytics import YOLO
# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v8/yolov8.yaml"
#数据集配置文件
data_yaml_path = 'datasets/Data/data.yaml'
#预训练模型
pre_model_name = 'yolov8n.pt'

if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(data=data_yaml_path,
                          device=0,
                          workers=0,
                          epochs=200,
                          patience=30,
                          batch=32,
                          name='train_v8')