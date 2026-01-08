import torch
import torch.nn as nn

class SimpleDetector(nn.Module):
    def __init__(self, S=4, B=1, C=1):
        super(SimpleDetector, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 17, 1024),
            nn.ReLU(),
            nn.Linear(1024, S * S * (B * 5 + C))
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)

        # 添加 sigmoid 激活以保证输出概率合法
        x[..., 0:2] = torch.sigmoid(x[..., 0:2])     # x, y in cell
        x[..., 4] = torch.sigmoid(x[..., 4])         # objectness
        x[..., 5:] = torch.sigmoid(x[..., 5:])       # class prob (tumor)

        return x
