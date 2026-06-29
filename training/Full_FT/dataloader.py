from torch.utils.data import DataLoader
from itertools import cycle


#####################################################################
# get_old_datasets: data from [MNIST + EMNIST（digits） + SVHN]
# get_new_datasets: data for FFT [could be different new data sources]
# train_set : val_set : test_set = 90:5:5
#####################################################################
get_old_datasets = None 
get_new_datasets = None


########################################
# 
# tool function： get_dataloaders
# 
########################################
def get_dataloaders(args):
    
    sample_ratio = args.sample_ratio
    batch_size = args.batch_size

    r_n, r_o = map(int, sample_ratio.split(':'))
    base_size = batch_size // (r_n + r_o)
    new_batch_size = base_size * r_n
    old_batch_size = base_size * r_o

    train_dataset, val_dataset, test_dataset = get_old_datasets()
    old_train_dataset, old_val_dataset, old_test_dataset = get_old_datasets()
    
    new_train_loader = DataLoader(train_dataset, batch_size=new_batch_size, shuffle=True, num_workers=2, pin_memory=False)
    new_val_loader = DataLoader(val_dataset, batch_size=new_batch_size, shuffle=False, num_workers=2, pin_memory=False)
    new_test_loader = DataLoader(test_dataset, batch_size=new_batch_size, shuffle=False, num_workers=2, pin_memory=False)

    old_train_loader = DataLoader(old_train_dataset, batch_size=old_batch_size, shuffle=True, num_workers=2, pin_memory=False)
    old_val_loader = DataLoader(old_val_dataset, batch_size=old_batch_size, shuffle=False, num_workers=2, pin_memory=False)
    old_test_loader = DataLoader(old_test_dataset, batch_size=old_batch_size, shuffle=False, num_workers=2, pin_memory=False)

    if len(new_train_loader) >= len(old_train_loader):
        old_train_loader = cycle(old_train_loader)
    else:
        new_train_loader = cycle(new_train_loader)

    return new_train_loader, new_val_loader, new_test_loader, old_train_loader, old_val_loader, old_test_loader
    















