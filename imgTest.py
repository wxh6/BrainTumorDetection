#coding:utf-8
from ultralytics import YOLO
import cv2

# 所需加载的模型目录
path = 'runs/detect/train_v82/weights/best.pt'
# 需要检测的图片地址
img_path ="testCNN/volume_4_slice_59_jpg.rf.6f76df3e1d6e9e8bab3d0a144f4817d0.jpg"
# 加载预训练模型
# conf	0.25	object confidence threshold for detection
# iou	0.7	intersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')


# 检测图片
results = model(img_path,conf=0.5,iou=0.1)
print(results)
res = results[0].plot()
res = cv2.resize(res,dsize=None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
cv2.imshow("YOLOv8 Detection", res)
cv2.waitKey(0)