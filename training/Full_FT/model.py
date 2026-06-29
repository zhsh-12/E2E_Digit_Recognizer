import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential() # Default identity mapping: identity = x
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Digit_recognizer(nn.Module):
    """Digit_recognizer with ResNet"""
    def __init__(self, dropout_ratio=0.3):
        super().__init__()
        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Residual stage 1: 32 in, 32 out, 2x downsampling
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            nn.MaxPool2d(2) # Size halved: 28→14
        )
        # Residual stage 2: double channels (32→64), downsampling
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            nn.MaxPool2d(2) # Size halved: 14→7
        )
        # Residual stage 3: double channels (64→128), no downsampling
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=1) # Size stays 7×7
        )
        # Global average pooling: compress each feature map to a single value (7×7→1×1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier: FC → ReLU → Dropout → FC output 10 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
