from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, DataLoader, Subset
from pathlib import Path

# 本地数据路径
base_dir = (Path(__file__).resolve()).parent.parent
data_dir = base_dir / "data_for_training"


########################################
# 针对eval数据集：定义transform
# EMNIST数据做特别的方向修正处理
# MNIST\EMNIST做三通道相等的灰度处理，SVHN保持RGB状态
########################################
#处理MNIST\EMNIST数据/三通道相等的灰度处理
def get_gray_transforms():
    base = [
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3)
    ]   
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ] 
    eval_transform = transforms.Compose(base + normalize)
    return eval_transform

#单独处理SVHN数据/RGB输入
def get_rgb_transforms():
    base = [
        transforms.Resize((28, 28))
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
    eval_transform = transforms.Compose(base + normalize)
    return eval_transform

#特别处理emnist数据集：旋转+翻转，修正数据因储存格式问题产生的错误
class EMNIST:
    def __call__(self, img):
        img = TF.rotate(img, -90)
        img = TF.hflip(img)
        return img

########################################
# 
# 组装函数： get_calibration_dataloader
# 
########################################
def get_calibration_dataloader(mnist_samples=40, emnist_samples=40,svhn_samples=52, batch_size=1):
    #获取transform方法
    gray_eval_tf = get_gray_transforms()
    emnist_eval_tf = transforms.Compose([
        EMNIST(),
        gray_eval_tf
    ])
    rgb_eval_tf = get_rgb_transforms()
    
    # 加载 MNIST
    test_mnist = datasets.MNIST(root=data_dir, train=False, download=False, transform=gray_eval_tf)
    mnist_subset = Subset(test_mnist, range(mnist_samples))

    #加载 EMNIST（digits）
    test_emnist = datasets.EMNIST(root=data_dir, split="digits", train=False, download=False, transform=emnist_eval_tf)
    emnist_subset = Subset(test_emnist, range(emnist_samples))

    #加载 SVHN
    test_svhn = datasets.SVHN(root=data_dir, split="test", download=False,transform=rgb_eval_tf)
    svhn_subset = Subset(test_svhn, range(svhn_samples))

    #混合数据集
    mixed_dataset = ConcatDataset([
        mnist_subset,
        emnist_subset,
        svhn_subset
    ])

    #构建dataloader
    return DataLoader(mixed_dataset, batch_size=batch_size, shuffle=False)
    















