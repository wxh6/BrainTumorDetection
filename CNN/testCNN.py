import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from CNN_model import SimpleDetector  # 引用前面写的模型

# 参数
S, B, C = 4, 1, 1
IMG_SIZE = (132, 139)
THRESHOLD = 0.3  # objectness 阈值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = SimpleDetector(S=S, B=B, C=C).to(DEVICE)
model.load_state_dict(torch.load("C:/Users/11729/Desktop/ultralytics-main/checkpoints/detector_epoch100.pt", map_location=DEVICE))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# 测试图像路径
test_dir = 'C:/Users/11729/Desktop/ultralytics-main/testCNN'
test_images = sorted(os.listdir(test_dir))

# 推理函数
def detect_and_visualize(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)[0]  # shape: [S, S, B*5+C]

    output = output.cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    H, W = image.size

    for i in range(S):
        for j in range(S):
            cell = output[j, i]
            conf = cell[4]
            if conf > THRESHOLD:
                x_cell, y_cell, w_cell, h_cell = cell[0:4]
                x = (i + x_cell) / S * W
                y = (j + y_cell) / S * H
                w = w_cell * W
                h = h_cell * H
                x_min = x - w / 2
                y_min = y - h / 2
                rect = patches.Rectangle((x_min, y_min), w, h,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, f"tumor {conf:.2f}", color="red")

    plt.title(f"Prediction: {os.path.basename(image_path)}")
    plt.axis("off")
    plt.show()

# 批量测试图像
for img_file in test_images:
    img_path = os.path.join(test_dir, img_file)
    detect_and_visualize(img_path)
