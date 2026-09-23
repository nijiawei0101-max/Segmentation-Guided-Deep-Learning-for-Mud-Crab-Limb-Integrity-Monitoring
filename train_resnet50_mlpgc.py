

import os, random, shutil, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
from sklearn.manifold import TSNE


DATA_DIR = r"D:\NJW\Crab\classification\dataset"
OUT_DIR  = "inference_result/train_resnet50_mlpgc_plus_A"
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 30
MIXUP_EPOCHS = int(EPOCHS * 0.4)   
FREEZE_EPOCHS = 0
TTA = False
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]
SEED = 42
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/models", exist_ok=True)
os.makedirs(f"{OUT_DIR}/classified_images", exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def compute_flops_g(model: nn.Module, img_size=IMG_SIZE):
    try:
        from thop import profile  # pip install thop
        dummy = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception as e:
        print(f"[Warn] FLOPs compute failed (skip): {e}")
        return None


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=(2,3), keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class MLPGCBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.grn  = GlobalResponseNorm(ch)
        self.conv1= nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1  = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2= nn.Conv2d(ch, ch, 1)
        self.bn2  = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = self.grn(x)
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + x)

class ResNetWithMLPGC(nn.Module):
    def __init__(self, num_classes=2, dropout=0.6):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = nn.Sequential(base.layer3, MLPGCBlock(1024))
        self.layer4 = nn.Sequential(base.layer4, MLPGCBlock(2048))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


def build_loaders(data_dir, batch_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=min(8, os.cpu_count()), pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=min(8, os.cpu_count()), pin_memory=True)
    return train_loader, val_loader, train_ds, val_ds

# --------------------- 训练辅助 ---------------------
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@torch.no_grad()
def tta_forward(model, images):
    if not TTA:
        return model(images)
    logits = model(images)
    flipped = torch.flip(images, dims=[3])
    logits_flipped = model(flipped)
    return (logits + logits_flipped) / 2.0

def freeze_until_layer4(model, freeze=True):
    modules = [model.stem, model.layer1, model.layer2, model.layer3]
    for m in modules:
        for p in m.parameters():
            p.requires_grad = not freeze


@torch.no_grad()
def evaluate_full(model, loader, criterion, classes):
    model.eval()
    total, correct = 0.0, 0
    all_preds, all_labels, all_logits = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = tta_forward(model, imgs)
        loss = criterion(logits, labels)
        total += loss.item() * imgs.size(0)

        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(logits.detach().cpu().numpy())

    val_loss = total / len(loader.dataset)
    val_acc  = correct / len(loader.dataset)
    val_f1   = f1_score(all_labels, all_preds, average="macro")
    val_rec  = recall_score(all_labels, all_preds, average="macro")
    return val_loss, val_acc, val_f1, val_rec, np.array(all_preds), np.array(all_labels), np.array(all_logits)


def main():
    train_loader, val_loader, train_ds, val_ds = build_loaders(DATA_DIR, BATCH_SIZE)
    model = ResNetWithMLPGC(NUM_CLASSES, dropout=0.6).to(device)


    counts = np.bincount(train_ds.targets, minlength=NUM_CLASSES)
    class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)


    params_m = count_params_m(model)
    flops_g  = compute_flops_g(model, IMG_SIZE)
    print(f"[Model] Params: {params_m:.3f} M | FLOPs@{IMG_SIZE}: {('%.3f G' % flops_g) if flops_g is not None else 'N/A'}")


    if FREEZE_EPOCHS > 0:
        freeze_until_layer4(model, freeze=True)
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()

    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_path = 0.0, f"{OUT_DIR}/models/best_model.pth"
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    val_f1s, val_recs = [], []


    csv_file = os.path.join(OUT_DIR, "metrics.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_loss","train_acc","val_acc","f1_macro","recall_macro","params_M","FLOPs_G"])

    for epoch in range(EPOCHS):

        if FREEZE_EPOCHS > 0 and epoch == FREEZE_EPOCHS:
            freeze_until_layer4(model, freeze=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch)


        model.train()
        total_loss, correct, n = 0.0, 0, 0
        use_mixup = (epoch < MIXUP_EPOCHS)

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_mixup:
                x, ya, yb, lam = mixup_data(imgs, labels, alpha=0.4)
                logits = model(x)
                loss = mixup_criterion(criterion, logits, ya, yb, lam)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            if not use_mixup:  # 仅非 mixup 阶段统计训练准确率
                correct += (logits.argmax(1) == labels).sum().item()
                n += imgs.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc  = (correct / n) if n > 0 else float("nan")
        train_losses.append(train_loss); train_accs.append(train_acc)

        # ---------- 验证（当轮） ----------
        val_loss, val_acc, val_f1, val_rec, all_preds, all_labels, all_logits = \
            evaluate_full(model, val_loader, criterion, val_ds.classes)
        val_losses.append(val_loss); val_accs.append(val_acc)
        val_f1s.append(val_f1); val_recs.append(val_rec)
        scheduler.step()

        print(f"Epoch {epoch+1:02d} | TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} "
              f"| ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | F1 {val_f1:.4f} | Recall {val_rec:.4f}")

        # 记录到 CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{train_acc:.6f}", f"{val_acc:.6f}",
                             f"{val_f1:.6f}", f"{val_rec:.6f}",
                             f"{params_m:.3f}", f"{flops_g:.3f}" if flops_g is not None else ""])

        # 保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    # ---------- 用“最佳权重”做最终评估 ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    best_loss, best_acc, best_f1, best_rec, all_preds, all_labels, all_logits = \
        evaluate_full(model, val_loader, criterion, val_ds.classes)

    # ---------- 可视化 ----------
    # Loss/Acc 曲线
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(train_losses, label="Train Loss"); plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2); plt.plot(train_accs, label="Train Acc"); plt.plot(val_accs, label="Val Acc")
    plt.legend(); plt.title("Accuracy"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/loss_acc.png")

    # 混淆矩阵（最佳权重）
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=val_ds.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Best)")
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png")

    # t-SNE（最佳权重）
    try:
        feats2d = TSNE(n_components=2, random_state=SEED).fit_transform(np.array(all_logits))
        plt.figure(figsize=(6,5))
        labels_np = np.array(all_labels)
        for lbl in np.unique(labels_np):
            idx = labels_np == lbl
            plt.scatter(feats2d[idx,0], feats2d[idx,1], label=str(val_ds.classes[lbl]), alpha=0.6)
        plt.legend(); plt.title("t-SNE (Best)")
        plt.savefig(f"{OUT_DIR}/tsne.png")
    except Exception as e:
        print(f"[Warn] TSNE failed (skip): {e}")


    val_img_paths = [s[0] for s in val_ds.samples]
    for i in range(min(100, len(all_preds))):
        true_label = val_ds.classes[all_labels[i]]
        pred_label = val_ds.classes[all_preds[i]]
        dst = f"{OUT_DIR}/classified_images/{i}_{true_label}_pred_{pred_label}.jpg"
        try:
            shutil.copy(val_img_paths[i], dst)
        except Exception:
            pass


    final_txt = os.path.join(OUT_DIR, "final_metrics.txt")
    with open(final_txt, "w") as f:
        f.write(
            "=== Final (Best Checkpoint) ===\n"
            f"Params (M): {params_m:.3f}\n"
            f"FLOPs@{IMG_SIZE} (G): {flops_g:.3f}\n" if flops_g is not None else
            f"Params (M): {params_m:.3f}\nFLOPs@{IMG_SIZE} (G): N/A\n"
        )
        f.write(
            f"Val Loss: {best_loss:.6f}\n"
            f"Val Acc : {best_acc:.6f}\n"
            f"F1 (macro): {best_f1:.6f}\n"
            f"Recall (macro): {best_rec:.6f}\n"
        )

    print(
        f"✅ Done. Best@Val | Acc: {best_acc:.4f} | F1: {best_f1:.4f} | Recall: {best_rec:.4f} | "
        f"Params: {params_m:.3f}M | FLOPs@{IMG_SIZE}: {('%.3fG' % flops_g) if flops_g is not None else 'N/A'}\n"
        f"Model saved to {best_path}\n"
        f"Per-epoch metrics -> {csv_file}\n"
        f"Final metrics -> {final_txt}"
    )

if __name__ == "__main__":
    main()
