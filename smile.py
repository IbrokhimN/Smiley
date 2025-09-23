import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm


# -------------------------------
# Конфиг
# -------------------------------
@dataclass
class Config:
    seed: int = 42
    dataset_size: int = 5000
    batch_size: int = 32
    lr: float = 0.001
    max_epochs: int = 50
    train_split: float = 0.8
    model_path: str = "smiley_model_varied.pth"


# -------------------------------
# Инициализация
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -------------------------------
# Датасет
# -------------------------------
class SmileyDataset(Dataset):
    def __init__(self, num_samples: int = 5000):
        self.num_samples = num_samples
        self.data, self.labels = self._generate_dataset(num_samples)

    def _generate_dataset(self, num_samples):
        data_list, labels_list = [], []
        for _ in range(num_samples):
            img, label = self._generate_sample()
            data_list.append(img)
            labels_list.append(label)

        data = torch.tensor(np.array(data_list), dtype=torch.float32).unsqueeze(1)  # [N,1,100,100]
        labels = torch.tensor(np.array(labels_list), dtype=torch.long)
        return data, labels

    def _generate_sample(self):
        img = np.zeros((100, 100), dtype=np.float32)

        offset_x = random.randint(10, 40)
        offset_y = random.randint(10, 40)

        if random.random() > 0.5:
            # Смайлик :)
            eye_y = offset_y
            eye_x_left = offset_x
            eye_x_right = offset_x + 40

            img[eye_y, eye_x_left] = 1
            img[eye_y, eye_x_right] = 1

            for i in range(eye_x_left, eye_x_right):
                j = int(offset_y + 30 + 10 * np.sin((i - eye_x_left) / 40 * np.pi))
                img[j, i] = 1

            label = 1
        else:
            # Шум / грусть
            for _ in range(50):
                x, y = random.randint(0, 99), random.randint(0, 99)
                img[y, x] = 1
            label = 0

        # Шум
        for _ in range(random.randint(0, 5)):
            x, y = random.randint(0, 99), random.randint(0, 99)
            img[y, x] = 1

        return img, label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# -------------------------------
# Модель
# -------------------------------
class SmileyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 25 * 25, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------------
# Обучение
# -------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(dataloader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


# -------------------------------
# Визуализация примеров
# -------------------------------
def show_examples(dataset, n=6):
    plt.figure(figsize=(10, 3))
    for i in range(n):
        img, label = dataset[i]
        plt.subplot(1, n, i + 1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title("Smiley" if label == 1 else "Noise")
        plt.axis("off")
    plt.show()


# -------------------------------
# Main
# -------------------------------
def main(cfg: Config):
    set_seed(cfg.seed)

    dataset = SmileyDataset(num_samples=cfg.dataset_size)

    # Train/Test split
    train_size = int(cfg.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # Показ примеров
    show_examples(dataset)

    # Модель
    model = SmileyCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Обучение
    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch:02d} | "
              f"Train: loss={train_loss:.4f}, acc={train_acc*100:.2f}% | "
              f"Test: loss={test_loss:.4f}, acc={test_acc*100:.2f}%")

        if test_acc == 1.0:
            torch.save(model.state_dict(), cfg.model_path)
            print(f"✅ Perfect accuracy reached! Model saved at {cfg.model_path}")
            break


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
