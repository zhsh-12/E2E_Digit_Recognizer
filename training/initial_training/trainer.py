import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import time, os, shutil
from pathlib import Path
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


########################################
#
# device setting
#
########################################

def set_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

########################################
#
# training for 1 epoch 
#
########################################

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train() 

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad() 
        outputs = model(x) 
        loss = criterion(outputs, y) 
        loss.backward() 
        optimizer.step() 

        total_loss += loss.item() * x.size(0) 
        preds = outputs.argmax(dim=1) 
        total_correct += (preds == y).sum().item() 
        total_samples += x.size(0)
    
    return total_loss/total_samples, total_correct/total_samples 

########################################
#
# evaluation for 1 epoch
#
########################################

def evaluate(model, loader, criterion, device):
    model.eval() 

    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    source_correct = defaultdict(int)
    source_total = defaultdict(int)
    
    all_preds = []
    all_labels = []
    all_sources = []

    with torch.no_grad():
        for x, y, source in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
            for i in range(len(source)):
                s = source[i].item()
                source_total[s] += 1
                if preds[i].item() == y[i].item():
                    source_correct[s] += 1
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            all_sources.extend(source.detach().cpu().numpy())

    overall_acc = total_correct / total_samples
    source_acc = {
        s: source_correct[s] / source_total[s] for s in source_total
    } 

    return total_loss / total_samples, overall_acc, source_acc, all_preds, all_labels,all_sources

########################################
#
# save function：to save training state
#
########################################

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), 
        "scheduler": scheduler.state_dict(), 
        "epoch": epoch, 
        "best_val_acc": best_val_acc 
    }, path)

########################################
#
# loading function：to restore training when acidentally stop
#
########################################

def load_checkpoint(model, optimizer, scheduler, path, device):
    if path is None or not os.path.exists(path):
        print("No checkpoint found, start from scratch")
        return 1, 0.0  #starting from Epoch 1
    print(f"Loading checkpoint from {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    return start_epoch, best_val_acc


########################################
#
# main trainer
#
########################################

def train_model(
        model, train_loader, val_loader, device, epochs=20, lr=1e-3, patience=5,
        resume_path=None,  log_path=None, output_dir=None
    ):

    if output_dir is None:
        raise ValueError("output_dir must be provided")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if log_path is None:
        raise ValueError("log_path must be provided")
    log_path = Path(log_path)
    if log_path.exists():
        shutil.rmtree(log_path) 
    log_path.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / 'best.pth'
    last_path = output_dir / 'last.pth'
    
    #Step 1: method setting
    model.to(device) 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3, min_lr=1e-6)
    writer = SummaryWriter(log_dir=str(log_path))

    #Step 2: restore training
    start_epoch, best_val_acc = load_checkpoint(model, optimizer, scheduler, resume_path, device)
    print("Current LR:", optimizer.param_groups[0]["lr"]) 

    #Step 3: variable setting
    no_improve_count = 0

    metrics = {
        'train_loss': None,
        'val_loss': None,
        'train_acc': None,
        'val_acc': None
    }

    source_map = {0: "MNIST", 1: "EMNIST", 2: "SVHN"}

    #Step 4: model training
    for epoch in range(start_epoch, epochs+1):
        start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_source_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.perf_counter() - start_time

        print(f"\nEpoch [{epoch}/{epochs}] - {elapsed:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        print("Val per-domain accuracy:")
        for k in sorted(val_source_acc.keys()):
            source = source_map.get(k, f"Unknown-{k}")
            print(f"{source}: {val_source_acc[k]:.4f}")
        
        metrics['train_loss'] = train_loss
        metrics['val_loss'] = val_loss
        metrics['train_acc'] = train_acc
        metrics['val_acc'] = val_acc

        #update lr
        scheduler.step(val_loss)

        #save best model
        if val_acc > best_val_acc + 1e-4: 
            best_val_acc = val_acc 
            no_improve_count = 0 
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, best_path)
            print("✅Saved best model")
        else:
            no_improve_count += 1
            print(f"⚠️ No improvement for {no_improve_count} epoch(s)")
        
        #save newest model
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, last_path)
        
        #TensorBoard logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)    
        for k, v in val_source_acc.items():
            source = source_map.get(k, f"Unknown-{k}")
            writer.add_scalar(f"Accuracy/Val/{source}", v, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        #Early Stopping
        if no_improve_count >= patience:
            print(f"⚠️Early stopping triggered (patience={patience})")
            break

    writer.close()

    print(f"\nBest Val Acc: {best_val_acc:.4f}")

    return metrics['train_loss'], metrics['val_loss'], metrics['train_acc'], metrics['val_acc']

###################################################################
#
# tool function： normalized confusion_matrix & visualization
#
###################################################################

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0, 
                xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def normalized_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds, labels=np.arange(10)) 
    cm = cm.astype(np.float32)
    row_nums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_nums, where=row_nums != 0)
    return cm


########################################
#
# evaluate model[test_set]
#
########################################

def test_model(model, test_loader, device, png_path):
    model.to(device) 

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    test_loss, test_acc, test_source_acc, all_preds, all_labels, all_sources = evaluate(model, test_loader, criterion, device)
    source_map = {0: "MNIST", 1: "EMNIST", 2: "SVHN"}

    print("\n=====Test Results=====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print("Test per-domain accuracy:")
    for k in sorted(test_source_acc.keys()):
        source = source_map.get(k, f"Unknown-{k}")
        print(f"{source}: {test_source_acc[k]:.4f}")
    
    # Overall normalized confusion matrix
    overall_cm = normalized_confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        overall_cm,
        title = "Overall Normalized Confusion Matrix",
        save_path = png_path / "overall_normalized_confusion_matrix.png"
    )
    print("\n✅ Saved overall normalized confusion matrix")

    #Pre-domain confusion matrix
    unique_sources = sorted(set(all_sources))
    for source_id in unique_sources:
        domain_preds = []
        domain_labels = []
        for pred, label, source in zip(all_preds, all_labels, all_sources):
            if source == source_id:
                domain_preds.append(pred)
                domain_labels.append(label)
        domain_cm = normalized_confusion_matrix(domain_labels, domain_preds)
        domain_name = source_map.get(int(source_id), f"Unknown-{source_id}")
        save_path = f"{domain_name.lower()}_normalized_confusion_matrix.png"
        plot_confusion_matrix(
            domain_cm,
            title = f"{domain_name} Normalized Confusion Matrix",
            save_path = png_path / save_path
        )
        print(f"✅ Saved {domain_name} normalized confusion matrix")

    return test_loss, test_acc, test_source_acc

