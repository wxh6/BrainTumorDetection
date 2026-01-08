import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from CNN_model import SimpleDetector


# ========== 模型定义 ==========
# class SimpleDetector(nn.Module):
#     def __init__(self, S=4, B=1, C=1):
#         super(SimpleDetector, self).__init__()
#         self.S = S
#         self.B = B
#         self.C = C
#
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 16 * 17, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, S * S * (B * 5 + C))
#         )
#
#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.fc(x)
#         return x.view(-1, self.S, self.S, self.B * 5 + self.C)

# ========== 数据集 ==========
class TumorDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=(132, 139), S=4, B=1, C=1):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        label_path = os.path.join(self.label_dir, filename.replace(".jpg", ".txt").replace(".jpeg", ".txt"))

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label_tensor = torch.zeros((self.S, self.S, self.B * 5 + self.C))
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    grid_x = int(x * self.S)
                    grid_y = int(y * self.S)

                    x_cell = x * self.S - grid_x
                    y_cell = y * self.S - grid_y
                    w_cell = w
                    h_cell = h

                    label_tensor[grid_y, grid_x, 0:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell, 1])
                    label_tensor[grid_y, grid_x, 5] = 1

        return image, label_tensor

# ========== 训练主程序 ==========
def train():
    # 参数设置
    S, B, C = 4, 1, 1
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 0.001

    train_dataset = TumorDetectionDataset(
        image_dir='datasets/Data/train/images',
        label_dir='datasets/Data/train/labels',
        S=S, B=B, C=C
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleDetector(S=S, B=B, C=C)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f"checkpoints/detector_epoch{epoch+1}.pt")

if __name__ == '__main__':
    train()
