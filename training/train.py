import json
import joblib
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.cuda.amp import GradScaler, autocast
import argparse

# ----------------------------
# Dataset
# ----------------------------
# Build Dataset Class
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_dir, label_names, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.label_names = label_names
        self.transform = transform
        self.labels = dataframe[self.label_names].values
        self.image_names = dataframe["Image Index"].values

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx): 
        image_path = os.path.join(self.image_dir, self.image_names[idx]) 
        image = Image.open(image_path).convert("RGB")                              
        label = torch.FloatTensor(self.labels[idx])
    
        if self.transform: 
            image = self.transform(image)

        return image, label

# ----------------------------
# Model
# ----------------------------
def get_model(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ----------------------------
# Training
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device): 
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach().item()

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device): 
    model.eval()
    total_loss = 0
    all_labels = []
    all_outputs = []
    
    for images, labels in tqdm(loader, desc="Validating"): 
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.detach().item()
    
        all_labels.append(labels.cpu())
        all_outputs.append(torch.sigmoid(outputs).cpu())

    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)

    #AUROC per class 
    aurocs = []
    for i in range(all_labels.shape[1]): 
        try:
            auc = roc_auc_score(all_labels[:,i], all_outputs[:,i])
            aurocs.append(auc)
        except ValueError:
            aurocs.append(float("nan")) # if only one class present

    macro_auroc = np.nanmean(aurocs) # average across valid classes
    
    return total_loss / len(loader), aurocs, macro_auroc

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/sample_labels.csv')
    parser.add_argument('--image_dir', type=str, default='data/images')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint_best.pth')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    df["Finding Labels"] = df["Finding Labels"].replace("No Finding", "")
    df["labels"] = df["Finding Labels"].str.split("|")
    df["labels"] = df["labels"].apply(lambda x: [label for label in x if label != ""])
    global label_names
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df["labels"])
    label_names = mlb.classes_
    for i, name in enumerate(label_names):
        df[name] = labels[:, i]
    joblib.dump(mlb, "mlb_classes.pkl")

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=22)
    for train_idx, val_idx in mskf.split(df, labels):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        break

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = ChestXrayDataset(train_df, args.image_dir, label_names, transform=train_transform)
    val_ds = ChestXrayDataset(val_df, args.image_dir, label_names, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = get_model(num_classes=len(label_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    best_macro_auroc = 0
    best_epoch = 0
    no_improve_epochs = 0

    train_losses, val_losses, val_aurocs = [], [], []

    start_epoch = 0
    best_macro_auroc = 0

    if args.resume:
        if os.path.exists("checkpoint_best.pth"):
            print("Resuming from checkpoint_best.pth...")
            checkpoint = torch.load("checkpoint_best.pth", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_losses = checkpoint['train_loss']
            val_losses = checkpoint['val_loss']
            val_aurocs = checkpoint['val_aurocs']
            start_epoch = checkpoint['epoch']
            best_macro_auroc = checkpoint['best_macro_auroc']
        else:
            print("Resume flag set but no checkpoint found.")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_auc, macro = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aurocs.append(val_auc)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro AUROC: {macro:.4f}")

        #save latest model every epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_aurocs': val_aurocs,
            'best_macro_auroc': best_macro_auroc
        }, "checkpoint_latest.pth")

        #save best model
        if macro > best_macro_auroc:
            best_macro_auroc = macro
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_aurocs': val_aurocs,
                'best_macro_auroc': best_macro_auroc
            }, "checkpoint_best.pth")
            print(f"✅ New best model saved (epoch {best_epoch})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    with open("training_log.json", "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_aurocs": val_aurocs
        }, f)

if __name__ == "__main__":
    main()
    