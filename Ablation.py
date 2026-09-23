# -*- coding: utf-8 -*-
# ResNet50_MLPGC_Plus with CSV logging (Precision/Recall/F1) and ablation switches

import os, random, shutil, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.manifold import TSNE

# --------------------- 设备 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- 解析参数（用于消融） ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    # bookkeeping
    p.add_argument('--tag', type=str, default='run')
    p.add_argument('--out_dir', type=str, default='D:/NJW/Crab/classification/Ablation Experiment/train_resnet50_mlpgc_plus_Abl')
    p.add_argument('--seed', type=int, default=42)

    # data
    p.add_argument('--data_dir', type=str, default='D:/NJW/Crab/classification/dataset')
    p.add_argument('--img_size', type=int, default=320)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--crop_min_scale', type=float, default=0.6)
    p.add_argument('--use_color', type=int, default=1)
    p.add_argument('--use_rot', type=int, default=1)
    p.add_argument('--use_erasing', type=int, default=1)

    # architecture
    p.add_argument('--use_mlpgc_l3', type=int, default=1)   # after layer3 (C4)
    p.add_argument('--use_mlpgc_l4', type=int, default=1)   # after layer4 (C5)
    p.add_argument('--dropout_p', type=float, default=0.6)

    # optimization & eval
    p.add_argument('--use_llrd', type=int, default=1)
    p.add_argument('--freeze_bn', type=int, default=1)
    p.add_argument('--label_smooth', type=float, default=0.10)
    p.add_argument('--use_class_weights', type=int, default=1)
    p.add_argument('--mixup_epochs_frac', type=float, default=0.40)  # 0.0 to disable
    p.add_argument('--tta', type=int, default=1)
    p.add_argument('--base_lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=5e-5)

    p.add_argument('--stage', type=int, choices=[1, 2, 3, 4, 5], default=5,
                   help='1:stem; 2:+layer1; 3:+layer2; 4:+layer3(+MLPGC); 5:+layer4(+MLPGC)')

    return p.parse_args()

# --------------------- 随机种子 ---------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --------------------- 模型（ResNet50 + MLPGC） ---------------------
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
    """
    支持按 stage 截断主干，并在 C4/C5 后可选插入 MLPGC。
    stage:
      1 = stem (conv1+bn1+relu+maxpool)
      2 = stage1 (3 blocks)
      3 = stage2 (4 blocks)
      4 = stage3 (6 blocks) + [可选] MLPGC(1024)
      5 = stage4 (3 blocks) + [可选] MLPGC(2048)
    """
    def __init__(self, num_classes=2, dropout=0.6, stage=5,
                 use_l3=True, use_l4=True):
        super().__init__()
        assert stage in [1,2,3,4,5]
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # --- stem ---
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)

        # --- 依 stage 决定是否启用后续层 ---
        self.layer1 = base.layer1 if stage >= 2 else nn.Identity()
        self.layer2 = base.layer2 if stage >= 3 else nn.Identity()

        if stage >= 4:
            self.layer3 = base.layer3 if not use_l3 else nn.Sequential(base.layer3, MLPGCBlock(1024))
        else:
            self.layer3 = nn.Identity()

        if stage >= 5:
            self.layer4 = base.layer4 if not use_l4 else nn.Sequential(base.layer4, MLPGCBlock(2048))
        else:
            self.layer4 = nn.Identity()

        # --- 输出通道随 stage 变化 ---
        if stage == 1:   feat_dim = 64    # stem 输出通道
        elif stage == 2: feat_dim = 256   # C2
        elif stage == 3: feat_dim = 512   # C3
        elif stage == 4: feat_dim = 1024  # C4
        else:            feat_dim = 2048  # C5

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)        # stem
        x = self.layer1(x)      # 3 blocks
        x = self.layer2(x)      # 4 blocks
        x = self.layer3(x)      # 6 blocks (+ MLPGC 可选)
        x = self.layer4(x)      # 3 blocks (+ MLPGC 可选)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


# --------------------- LLRD / 冻结 BN ---------------------
def freeze_bn_stats(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()            # freeze running mean/var
            m.requires_grad_(True)  # keep affine trainable

def param_groups_llrd(model: nn.Module, base_lr=3e-4, weight_decay=5e-5):
    groups = []
    def add(pg, p, name, lr):
        no_decay = (name.endswith(".bias") or ".bn" in name or "bn." in name or "downsample.1" in name)
        pg.append({"params":[p], "lr":lr, "weight_decay":0.0 if no_decay else weight_decay})
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("stem.", "layer1.")):
            lr = base_lr * 0.25
        elif name.startswith("layer2."):
            lr = base_lr * 0.50
        elif name.startswith("layer3."):
            lr = base_lr * 0.75
        else:
            lr = base_lr * 1.00
        add(groups, p, name, lr)
    return groups

# --------------------- 数据加载 ---------------------
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

def build_loaders(args):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(args.crop_min_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        *( [transforms.ColorJitter(0.3,0.3,0.3,0.1)] if args.use_color else [] ),
        *( [transforms.RandomRotation(10)] if args.use_rot else [] ),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
        *( [transforms.RandomErasing(p=0.25, scale=(0.02,0.2), ratio=(0.3,3.3))] if args.use_erasing else [] ),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=min(8, os.cpu_count()), pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=min(8, os.cpu_count()), pin_memory=True)
    return train_loader, val_loader, train_ds, val_ds

# --------------------- Mixup & TTA ---------------------
def mixup_data(x, y, alpha=0.4):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1.0
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@torch.no_grad()
def tta_forward(model, images, use_tta=True):
    if not use_tta:
        return model(images)
    logits = model(images)
    flipped = torch.flip(images, dims=[3])
    logits_flipped = model(flipped)
    return (logits + logits_flipped) / 2.0

# --------------------- 主流程 ---------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # 输出目录组织：out_dir/tag
    OUT_DIR = os.path.join(args.out_dir, args.tag)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/classified_images", exist_ok=True)

    # 数据
    train_loader, val_loader, train_ds, val_ds = build_loaders(args)

    # 模型
    # 模型
    use_l3_auto = bool(args.use_mlpgc_l3) if args.stage >= 4 else False
    use_l4_auto = bool(args.use_mlpgc_l4) if args.stage >= 5 else False
    model = ResNetWithMLPGC(
        num_classes=2,
        dropout=args.dropout_p,
        stage=args.stage,
        use_l3=use_l3_auto,
        use_l4=use_l4_auto
    ).to(device)

    if args.freeze_bn:
        freeze_bn_stats(model)

    # loss（类权重 + label smoothing）
    class_weights = None
    if args.use_class_weights:
        counts = np.bincount(train_ds.targets, minlength=2)
        class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smooth)

    # 优化器 & 调度器
    if args.use_llrd:
        params = param_groups_llrd(model, base_lr=args.base_lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(params)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixup 轮数
    MIXUP_EPOCHS = int(args.epochs * args.mixup_epochs_frac)

    # 统计
    best_acc = 0.0
    best_epoch = -1
    best_prec_bin = best_rec_bin = best_f1_bin = 0.0
    best_prec_mac = best_rec_mac = best_f1_mac = 0.0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # 正类索引：以“broken”为正类（若不存在，则默认 1）
    pos_label_idx = val_ds.class_to_idx.get('broken', 1)

    for epoch in range(args.epochs):
        # --------------- 训练 ---------------
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        use_mixup = (epoch < MIXUP_EPOCHS)

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
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

        # --------------- 验证（含逐样本损失，导出 Top-50） ---------------
        model.eval()
        val_total, val_correct = 0.0, 0
        all_preds, all_labels, all_feats = [], [], []
        val_details = []
        val_paths = [p for (p, _) in val_ds.samples]  # 保持 dataloader 顺序
        val_ptr = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = tta_forward(model, imgs, use_tta=bool(args.tta))
                loss = criterion(logits, labels)
                val_total += loss.item() * imgs.size(0)

                probs = torch.softmax(logits, dim=1)
                pred  = probs.argmax(1)

                # 记录
                val_correct += (pred == labels).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_feats.extend(logits.cpu().numpy())

                # 逐样本损失（用于 Top-50）
                per_sample_loss = F.cross_entropy(
                    logits, labels, weight=class_weights, label_smoothing=args.label_smooth, reduction='none'
                )
                bs = imgs.size(0)
                for j in range(bs):
                    path = val_paths[val_ptr]; val_ptr += 1
                    true_id = int(labels[j].cpu().item())
                    pred_id = int(pred[j].cpu().item())
                    conf    = float(probs[j, pred_id].cpu().item())
                    loss_val= float(per_sample_loss[j].cpu().item())
                    val_details.append({
                        "loss": loss_val, "path": path,
                        "true_id": true_id, "pred_id": pred_id, "conf": conf
                    })

        val_loss = val_total / len(val_loader.dataset)
        val_acc  = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss); val_accs.append(val_acc)
        scheduler.step()

        # 计算 Precision/Recall/F1（binary & macro）
        prec_bin = precision_score(all_labels, all_preds, average='binary', pos_label=pos_label_idx, zero_division=0)
        rec_bin  = recall_score(all_labels, all_preds, average='binary', pos_label=pos_label_idx, zero_division=0)
        f1_bin   = f1_score(all_labels, all_preds, average='binary', pos_label=pos_label_idx, zero_division=0)

        prec_mac = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec_mac  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_mac   = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1:02d} | TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} "
              f"| ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | F1(bin) {f1_bin:.4f}")

        # 保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_prec_bin, best_rec_bin, best_f1_bin = prec_bin, rec_bin, f1_bin
            best_prec_mac, best_rec_mac, best_f1_mac = prec_mac, rec_mac, f1_mac
            torch.save(model.state_dict(), f"{OUT_DIR}/models/best_model.pth")

        # === 导出本 epoch 验证集损失最大的 50 张 ===
        export_dir = os.path.join(OUT_DIR, "val_top_losses_50", f"ep{epoch+1:03d}")
        os.makedirs(export_dir, exist_ok=True)
        val_details.sort(key=lambda d: d["loss"], reverse=True)
        topK = val_details[:50]

        csv_path_top = os.path.join(export_dir, "val_top_losses_50.csv")
        with open(csv_path_top, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["rank","loss","true","pred","conf","path"])
            for r, d in enumerate(topK, 1):
                w.writerow([
                    r, f"{d['loss']:.6f}", val_ds.classes[d["true_id"]],
                    val_ds.classes[d["pred_id"]], f"{d['conf']:.4f}", d["path"]
                ])
        for r, d in enumerate(topK, 1):
            src = d["path"]; ext = os.path.splitext(src)[1]
            dst = os.path.join(
                export_dir,
                f"{r:03d}_loss{d['loss']:.4f}_true-{val_ds.classes[d['true_id']]}_"
                f"pred-{val_ds.classes[d['pred_id']]}_p{d['conf']:.3f}{ext}"
            )
            try: shutil.copy(src, dst)
            except Exception as e: print(f"[WARN] copy failed: {src} -> {dst}: {e}")

    # ---------- 可视化 ----------
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(train_losses, label="Train Loss"); plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2); plt.plot(train_accs, label="Train Acc"); plt.plot(val_accs, label="Val Acc")
    plt.legend(); plt.title("Accuracy"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/loss_acc.png")

    # 混淆矩阵（最后一轮）
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=val_ds.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png")

    # t-SNE
    feats2d = TSNE(n_components=2, random_state=args.seed).fit_transform(np.array(all_feats))
    plt.figure(figsize=(6,5))
    labels_np = np.array(all_labels)
    for lbl in np.unique(labels_np):
        idx = labels_np == lbl
        plt.scatter(feats2d[idx,0], feats2d[idx,1], label=str(val_ds.classes[lbl]), alpha=0.6)
    plt.legend(); plt.title("t-SNE")
    plt.savefig(f"{OUT_DIR}/tsne.png")

    # 复制前100张验证图，带预测
    val_img_paths = [s[0] for s in val_ds.samples]
    for i in range(min(100, len(all_preds))):
        true_label = val_ds.classes[all_labels[i]]
        pred_label = val_ds.classes[all_preds[i]]
        dst = f"{OUT_DIR}/classified_images/{i}_{true_label}_pred_{pred_label}.jpg"
        shutil.copy(val_img_paths[i], dst)

    # ---------- 写入总表 CSV（一次/每个 run） ----------
    summary_csv = os.path.join(args.out_dir, 'results_ablation.csv')  # 汇总在 out_dir 根目录
    is_new = not os.path.exists(summary_csv)
    with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                'tag','best_epoch',
                'img','bs','mlpgc_l3','mlpgc_l4','dropout',
                'llrd','freeze_bn','ls','class_w','mixup_frac','tta',
                'val_acc_best',
                'prec_bin_best','rec_bin_best','f1_bin_best',
                'prec_mac_best','rec_mac_best','f1_mac_best'
            ])
        w.writerow([
            os.path.basename(OUT_DIR), best_epoch,
            args.img_size, args.batch_size, args.use_mlpgc_l3, args.use_mlpgc_l4, args.dropout_p,
            args.use_llrd, args.freeze_bn, args.label_smooth, args.use_class_weights, args.mixup_epochs_frac, args.tta,
            f"{best_acc:.4f}",
            f"{best_prec_bin:.4f}", f"{best_rec_bin:.4f}", f"{best_f1_bin:.4f}",
            f"{best_prec_mac:.4f}", f"{best_rec_mac:.4f}", f"{best_f1_mac:.4f}",
        ])

    print(f"✅ Done. Best Val Acc: {best_acc:.4f} @ epoch {best_epoch}.")
    print(f"   CSV summary updated at: {summary_csv}")
    print(f"   Artifacts saved under:  {OUT_DIR}")

if __name__ == "__main__":
    main()
