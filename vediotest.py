
#yolo detect predict model=runs/detect/train/weight/best.pt source=./TestFiles/1.mp4 show=True
from ultralytics import YOLO

yolo=YOLO("models/best_v8.pt",task="detect")

result=yolo(source="TestFiles/1.mp4",save=True,show=True)