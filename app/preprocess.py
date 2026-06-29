import torch
from torchvision import transforms


def set_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


####################################################
#
# Data preprocessing:  grayscale/RGB/RGBA scenarios
#
####################################################
def transform(img):
    if img.mode == 'L': 
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale: r=g=b
            transforms.Resize((28, 28)),                  # Resize to 28x28
            transforms.ToTensor(),                        # Convert to Tensor, normalize pixels to [0,1]
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))          # Normalize to [-1, 1]
        ])
    elif img.mode == 'RGB': 
        transform = transforms.Compose([
            transforms.Resize((28, 28)),                  # Resize to 28x28
            transforms.ToTensor(),                        # Convert to Tensor, normalize pixels to [0,1]
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))          # Normalize to [-1, 1]
        ])
    elif img.mode == 'RGBA':
        img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((28, 28)),                  # Resize to 28x28
            transforms.ToTensor(),                        # Convert to Tensor, normalize pixels to [0,1]
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))          # Normalize to [-1, 1]
        ])
    else:
        raise ValueError(f"Unsupported image mode: {img.mode}")
    
    return transform(img)
