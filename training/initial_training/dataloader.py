import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, random_split, Dataset, DataLoader
from pathlib import Path


base_dir = (Path(__file__).resolve()).parent
data_dir = base_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)


########################################
#
# Download datasets
#
########################################
# MNIST
train_mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
test_mnist = datasets.MNIST(root=data_dir, train=False, download=True, transform=None)
mnist = ConcatDataset([train_mnist, test_mnist])


# EMNIST（digits）
train_emnist = datasets.EMNIST(root=data_dir, split="digits", train=True, download=True, transform=None)
test_emnist = datasets.EMNIST(root=data_dir, split="digits", train=False, download=True, transform=None)
emnist = ConcatDataset([train_emnist, test_emnist])


# SVHN
train_svhn = datasets.SVHN(root=data_dir, split="train", download=True, transform=None)
test_svhn = datasets.SVHN(root=data_dir, split="test", download=True,transform=None)
svhn = ConcatDataset([train_svhn, test_svhn])


########################################
#
# Extract 3 datasets in sample_ratio
#
########################################
def get_subset(dataset, target_size):
    return random_split(dataset, [target_size, len(dataset)-target_size], 
                        generator=torch.Generator().manual_seed(42))[0]

########################################
# 
# train_set : val_set : test_set = 90:5:5
# 
########################################
def split_dataset(dataset, train_ratio=0.9, val_ratio=0.05, seed=42):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))


########################################
# Define transform for train/eval
# special correction for EMNIST
# different transforms for grayscale/RGB
########################################
def get_gray_transforms(input_size, augmentation):
    base = [
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=3)
    ]
    
    aug_list = []
    aug_set = set(augmentation.split(',')) if augmentation else set()

    #Step 1: geometirc transform
    if 'rotate' in aug_set:
        aug_list.append(transforms.RandomRotation(10))
    if 'affine' in aug_set:
        aug_list.append(transforms.RandomAffine(
            degrees=15, 
            translate=(0.1,0.1), 
            scale=(0.9,1.1) 
        ))
    if 'flip' in aug_set:
        aug_list.append(transforms.RandomHorizontalFlip())
    if 'crop' in aug_set:
        aug_list.append(transforms.RandomResizedCrop(input_size,scale=(0.8, 1.0)))
    #Step 2: noise
    if 'blur' in aug_set:
        aug_list.append(transforms.GaussianBlur(kernel_size=3))
    #Step 3: color change
    if 'color' in aug_set:
        aug_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
    
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]

    #for train_set
    train_transform = transforms.Compose(base + aug_list + normalize)
    #for val_set/test_set
    eval_transform = transforms.Compose(base + normalize)

    return train_transform, eval_transform


def get_rgb_transforms(input_size, augmentation):
    base = [
        transforms.Resize((input_size, input_size))
    ]
    
    aug_list = []
    aug_set = set(augmentation.split(',')) if augmentation else set()

    #Step 1: geometirc transform
    if 'rotate' in aug_set:
        aug_list.append(transforms.RandomRotation(10))
    if 'affine' in aug_set:
        aug_list.append(transforms.RandomAffine(
            degrees=15, 
            translate=(0.1,0.1), 
            scale=(0.9,1.1) 
        ))
    if 'flip' in aug_set:
        aug_list.append(transforms.RandomHorizontalFlip())
    if 'crop' in aug_set:
        aug_list.append(transforms.RandomResizedCrop(input_size,scale=(0.8, 1.0)))
    #Step 2: noise
    if 'blur' in aug_set:
        aug_list.append(transforms.GaussianBlur(kernel_size=3))
    #Step 3: color change
    if 'color' in aug_set:
        aug_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
    
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]

    #for train_set
    train_transform = transforms.Compose(base + aug_list + normalize)
    #for val_set/test_set
    eval_transform = transforms.Compose(base + normalize)

    return train_transform, eval_transform

class EMNIST:
    def __call__(self, img):
        img = TF.rotate(img, -90)
        img = TF.hflip(img)
        return img

########################################
# Define class: transform + data source
# data source：mnist:0, emnist:1, svhn: 2
# 
########################################   
class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None, source_id=None):
        self.dataset = dataset
        self.transform = transform
        self.source_id = source_id
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        if self.source_id is not None:
            return x, y, self.source_id
        else:
            return x, y


########################################
# 
# tool function： get_dataloaders
# 
########################################
def get_dataloaders(args):
    
    input_size = args.input_size
    augmentation = args.augmentation
    sample_ratio = args.sample_ratio
    batch_size = args.batch_size
    
    r_m, r_e, r_s = map(int, sample_ratio.split(':'))
    base_size = min(len(mnist)//r_m, len(emnist)//r_e, len(svhn)//r_s)
    mnist_subset = get_subset(mnist, base_size * r_m)
    emnist_subset = get_subset(emnist, base_size * r_e)
    svhn_subset = get_subset(svhn, base_size * r_s)

    train_mnist, val_mnist, test_mnist = split_dataset(mnist_subset)
    train_emnist, val_emnist, test_emnist = split_dataset(emnist_subset)
    train_svhn, val_svhn, test_svhn = split_dataset(svhn_subset)

    Dataset_map = {"mnist": 0, "emnist": 1, "svhn": 2}

    gray_train_tf, gray_eval_tf = get_gray_transforms(input_size, augmentation)
    emnist_train_tf = transforms.Compose([
        EMNIST(),
        gray_train_tf
    ])
    emnist_eval_tf = transforms.Compose([
        EMNIST(),
        gray_eval_tf
    ])
    rgb_train_tf, rgb_eval_tf = get_rgb_transforms(input_size, augmentation)

    train_dataset = ConcatDataset([
        TransformDataset(train_mnist, gray_train_tf, Dataset_map['mnist']),
        TransformDataset(train_emnist, emnist_train_tf, Dataset_map['emnist']),
        TransformDataset(train_svhn, rgb_train_tf, Dataset_map['svhn'])
    ])
    val_dataset = ConcatDataset([
        TransformDataset(val_mnist, gray_eval_tf, Dataset_map['mnist']),
        TransformDataset(val_emnist, emnist_eval_tf, Dataset_map['emnist']),
        TransformDataset(val_svhn, rgb_eval_tf, Dataset_map['svhn'])
    ])
    test_dataset = ConcatDataset([
        TransformDataset(test_mnist, gray_eval_tf, Dataset_map['mnist']),
        TransformDataset(test_emnist, emnist_eval_tf, Dataset_map['emnist']),
        TransformDataset(test_svhn, rgb_eval_tf, Dataset_map['svhn'])
    ])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return train_loader, val_loader, test_loader
    















