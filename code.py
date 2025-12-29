import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import GradScaler, autocast
import torchvision
from torchvision import datasets, models
from torchvision.transforms import v2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. Global Settings & Reproducibility ---
CHECKPOINT_PATH = 'latest_checkpoint_scratch.pth'
BEST_MODEL_PATH = 'best_model_scratch.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # High performance

set_seed(42)

# --- 2. Advanced Data Augmentations ---
train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=256, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=20),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.2, scale=(0.02, 0.1))
])

val_transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def safe_ten_crop(img):
    crops = v2.functional.ten_crop(img, size=(224, 224))
    return torch.stack(crops)

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(256, antialias=True),
    v2.Lambda(safe_ten_crop), 
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. Improved Architecture ---
class GeM(nn.Module):
    """ Generalized Mean Pooling: Better for texture/medical feature extraction """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class ImprovedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet34(weights=None) # ResNet34 has more capacity than 18
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.gem = GeM()
        self.flatten = nn.Flatten()
        
        in_features = 512
        self.se = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gem(x)
        x = self.flatten(x)
        att = self.se(x)
        x = x * att
        return self.classifier(x)

# --- 4. MixUp/CutMix Utils ---
def apply_mixup_cutmix(X, y, alpha=1.0):
    if np.random.rand() > 0.5:
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(X.size(0)).to(DEVICE)
        mixed_X = lam * X + (1 - lam) * X[index, :]
        return mixed_X, y, y[index], lam
    else:
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(X.size(0)).to(DEVICE)
        W, H = X.size(2), X.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
        bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
        X[:, :, bbx1:bbx2, bby1:bby2] = X[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        return X, y, y[index], lam

# --- 5. Main Training Logic ---
def training():
    train_data = datasets.ImageFolder('Output/train', transform=train_transform)
    val_data   = datasets.ImageFolder('Output/val',   transform=val_transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    model = ImprovedResNet(num_classes=len(train_data.classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    lossfunc = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')
    
    epochs = 100
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    # RESUME INITIALIZATION
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔄 Resuming training from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        # Load RNG states for perfect consistency
        torch.set_rng_state(checkpoint['rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])
        print(f"✅ Successfully resumed at Epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        model.train()
        batch_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for X, y in loop:
            X, y = X.to(DEVICE), y.to(DEVICE)
            X_aug, y_a, y_b, lam = apply_mixup_cutmix(X, y)

            optimizer.zero_grad()
            with autocast('cuda'):
                pred = model(X_aug)
                loss = lam * lossfunc(pred, y_a) + (1 - lam) * lossfunc(pred, y_b)

            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer_stepped = scale_after >= scale_before
            
            if optimizer_stepped:
                scheduler.step()
            
            batch_losses.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), acc=((pred.argmax(dim=1) == y).float().mean()).item(), lr=current_lr)

        # Validation Phase
        model.eval()
        v_acc = []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                pred_v = model(X_v)
                acc = (pred_v.argmax(1) == y_v).float().mean().item()
                v_acc.append(acc)

        curr_val_acc = np.mean(v_acc) * 100
        history['train_loss'].append(np.mean(batch_losses))
        history['val_acc'].append(curr_val_acc)

        print(f"📊 Epoch {epoch+1}: Val Acc: {curr_val_acc:.2f}% | Best: {best_acc:.2f}%")

        # Save Best Model
        if curr_val_acc >= best_acc:
            best_acc = curr_val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        # SAVE FULL CHECKPOINT FOR RESUMPTION
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_acc': best_acc,
            'history': history,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state()
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

def testing():
    # Load model and class names
    train_data = datasets.ImageFolder('Output/train')
    class_names = train_data.classes
    model = ImprovedResNet(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    test_data = datasets.ImageFolder('Output/test', transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_t, y_t in tqdm(test_loader, desc="Testing"):
            X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
            # TenCrop logic
            bs, ncrops, c, h, w = X_t.size()
            outputs = model(X_t.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1) 
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(y_t.cpu().numpy())

    final_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    print(f"\n🚀 Final Test Accuracy: {final_acc:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.show()

if __name__ == "__main__":
    # training()
    testing()