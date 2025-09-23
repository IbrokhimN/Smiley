#!/usr/bin/env python3
"""
Smiley classification training script — improved, production-ready.
Features:
- on-the-fly dataset generation (or optional preloading)
- supports cuda / cpu / mps
- mixed precision (AMP) when available
- early stopping, scheduler, gradient clipping, checkpointing
- TensorBoard logging (optional)
"""
from dataclasses import dataclass
import argparse
import os
import time
import random
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    seed: int = 42
    dataset_size: int = 5000
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 50
    train_split: float = 0.8
    model_path: str = "smiley_model_best.pth"
    optimizer: str = "adam"     # 'adam' or 'sgd'
    use_scheduler: bool = True
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5
    grad_clip: float = 1.0
    early_stopping_patience: int = 6
    preload: bool = False       # pre-generate dataset into memory (like before)
    device_prefer: Optional[str] = None  # 'cuda', 'mps', 'cpu' or None for auto
    use_amp: bool = True        # mixed precision if available
    log_dir: str = "runs"
    save_every: int = 10        # periodic checkpoint saving (epochs)
    img_size: int = 100
    show_examples: bool = False


# -------------------------
# Utils
# -------------------------
def pick_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        p = prefer.lower()
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # may slow things down; optional
        torch.use_deterministic_algorithms(True)


# -------------------------
# Dataset (on-the-fly or preload)
# -------------------------
class SmileyDataset(Dataset):
    """Generates samples on-the-fly unless `preload=True`."""

    def __init__(self, num_samples: int = 5000, img_size: int = 100, preload: bool = False, seed: Optional[int] = None):
        self.num_samples = int(num_samples)
        self.img_size = int(img_size)
        self.preload = preload
        self.rng = random.Random(seed)  # separate RNG for determinism per-dataset
        if preload:
            self.data, self.labels = self._preload_all()
        else:
            self.data = None
            self.labels = None

    def _generate_sample(self) -> Tuple[np.ndarray, int]:
        s = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        offset_x = self.rng.randint(10, 40)
        offset_y = self.rng.randint(10, 40)

        if self.rng.random() > 0.5:
            # Smiley
            eye_y = offset_y
            eye_x_left = offset_x
            eye_x_right = offset_x + 40
            s[eye_y, eye_x_left] = 1.0
            s[eye_y, eye_x_right] = 1.0
            for i in range(eye_x_left, eye_x_right):
                j = int(offset_y + 30 + 10 * np.sin((i - eye_x_left) / 40 * np.pi))
                # boundary safety
                if 0 <= j < self.img_size:
                    s[j, i] = 1.0
            label = 1
        else:
            # Noise
            for _ in range(50):
                x = self.rng.randint(0, self.img_size - 1)
                y = self.rng.randint(0, self.img_size - 1)
                s[y, x] = 1.0
            label = 0

        # small random noise
        for _ in range(self.rng.randint(0, 5)):
            x = self.rng.randint(0, self.img_size - 1)
            y = self.rng.randint(0, self.img_size - 1)
            s[y, x] = 1.0

        return s, label

    def _preload_all(self):
        data_list, labels = [], []
        for i in range(self.num_samples):
            img, lab = self._generate_sample()
            data_list.append(img)
            labels.append(lab)
        data = torch.tensor(np.array(data_list), dtype=torch.float32).unsqueeze(1)
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        return data, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.preload:
            return self.data[idx], self.labels[idx]
        else:
            img, label = self._generate_sample()
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)


# -------------------------
# Model (slightly improved)
# -------------------------
class SmileyCNN(nn.Module):
    def __init__(self, img_size: int = 100):
        super().__init__()
        # conv stack
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 100 -> 50

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 50 -> 25
            nn.Dropout2d(0.1),
        )
        flattened = 64 * (img_size // 4) * (img_size // 4)
        self.classifier = nn.Sequential(
            nn.Linear(flattened, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# -------------------------
# Training utilities
# -------------------------
class EarlyStopping:
    """Simple early stopping by validation accuracy (higher is better)."""

    def __init__(self, patience: int = 6, mode: str = "max"):
        self.patience = int(patience)
        self.mode = mode
        self.best = None
        self.num_bad = 0

    def step(self, value: float) -> bool:
        """Return True if we should stop training"""
        if self.best is None:
            self.best = value
            self.num_bad = 0
            return False
        is_better = (value > self.best) if self.mode == "max" else (value < self.best)
        if is_better:
            self.best = value
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            return self.num_bad > self.patience


def get_optimizer(model: nn.Module, cfg: Config):
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    else:
        raise ValueError("Unknown optimizer: " + cfg.optimizer)


# -------------------------
# Train / Eval loop
# -------------------------
def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device,
                amp_scaler: Optional[torch.cuda.amp.GradScaler], cfg: Config) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if amp_scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            amp_scaler.scale(loss).backward()
            if cfg.grad_clip:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=total_loss / total, acc=100.0 * correct / total)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_targets.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc, np.concatenate(all_targets), np.concatenate(all_preds)


# -------------------------
# Utilities: plotting and saving
# -------------------------
def plot_examples(dataset: SmileyDataset, n: int = 6):
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2.5))
    for i in range(n):
        img, label = dataset[i]
        img_np = img.squeeze().numpy() if isinstance(img, torch.Tensor) else img.squeeze()
        axs[i].imshow(img_np, cmap="gray")
        axs[i].set_title("Smiley" if (label == 1 or (isinstance(label, torch.Tensor) and label.item() == 1)) else "Noise")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes=("Noise", "Smiley")):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# -------------------------
# Main
# -------------------------
def main(cfg: Config):
    # logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("smiley_train")
    device = pick_device(cfg.device_prefer)
    set_seed(cfg.seed)
    logger.info(f"Device: {device}; seed={cfg.seed}; preload={cfg.preload}")

    # dataset and dataloaders
    dataset = SmileyDataset(num_samples=cfg.dataset_size, img_size=cfg.img_size, preload=cfg.preload, seed=cfg.seed)
    train_size = int(cfg.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=2, pin_memory=(device.type == "cuda"))

    if cfg.show_examples:
        plot_examples(dataset, n=6)

    model = SmileyCNN(img_size=cfg.img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, cfg)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma) if cfg.use_scheduler else None
    amp_scaler = torch.cuda.amp.GradScaler() if (cfg.use_amp and device.type == "cuda") else None

    tb_writer = None
    if cfg.log_dir:
        tb_writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, time.strftime("%Y%m%d-%H%M%S")))

    early_stop = EarlyStopping(patience=cfg.early_stopping_patience)
    best_val_acc = 0.0

    for epoch in range(1, cfg.max_epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, amp_scaler, cfg)
        val_loss, val_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        duration = time.time() - start

        logger.info(f"Epoch {epoch:02d} | time {duration:.1f}s | train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}% | val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

        # tensorboard
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("Acc/train", train_acc, epoch)
            tb_writer.add_scalar("Acc/val", val_acc, epoch)
            tb_writer.flush()

        # scheduler step
        if scheduler:
            scheduler.step()

        # checkpoint & best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "cfg": cfg.__dict__,
            }, cfg.model_path)
            logger.info(f"Saved best model (val_acc={val_acc:.4f}) -> {cfg.model_path}")

        # periodic save
        if epoch % cfg.save_every == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }, f"{os.path.splitext(cfg.model_path)[0]}_epoch{epoch}.pth")

        # early stopping
        if early_stop.step(val_acc):
            logger.info(f"Early stopping triggered (no improvement for {cfg.early_stopping_patience} evals).")
            break

    # final report
    logger.info("Training finished. Best val_acc = %.4f" % best_val_acc)
    logger.info("\n" + classification_report(y_true, y_pred, target_names=["Noise", "Smiley"]))
    plot_confusion(y_true, y_pred)

    if tb_writer:
        tb_writer.close()


# -------------------------
# CLI
# -------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train SmileyCNN (improved)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_size", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--model_path", type=str, default="smiley_model_best.pth")
    p.add_argument("--optimizer", type=str, default="adam")
    p.add_argument("--use_scheduler", action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--preload", action="store_true")
    p.add_argument("--device", type=str, default=None, help="cuda/mps/cpu or leave empty for auto")
    p.add_argument("--show_examples", action="store_true")
    p.add_argument("--log_dir", type=str, default="runs")
    args = p.parse_args()
    return Config(
        seed=args.seed,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        train_split=args.train_split,
        model_path=args.model_path,
        optimizer=args.optimizer,
        use_scheduler=args.use_scheduler,
        use_amp=args.use_amp,
        preload=args.preload,
        device_prefer=args.device,
        show_examples=args.show_examples,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
