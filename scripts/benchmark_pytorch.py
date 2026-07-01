"""Benchmark PyTorch model inference speed."""
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
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


class DigitRecognizer(nn.Module):
    def __init__(self, dropout_ratio=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
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


def main():
    base_dir = Path(__file__).resolve().parent.parent
    device = 'cpu'
    
    print("Loading PyTorch model...")
    model = DigitRecognizer()
    checkpoint = torch.load(str(base_dir / 'trained_models/digit_recognizer.pth'), map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 28, 28)
    
    # Warmup
    print("Warming up...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    print("Benchmarking PyTorch FP32 (100 runs)...")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append((time.perf_counter() - start) * 1000)
    
    print(f'PyTorch FP32: mean={np.mean(times):.2f}ms, median={np.median(times):.2f}ms, std={np.std(times):.2f}ms, min={np.min(times):.2f}ms, max={np.max(times):.2f}ms')


if __name__ == '__main__':
    main()
