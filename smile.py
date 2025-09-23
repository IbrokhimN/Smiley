import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# -------------------------------
# Настройки
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -------------------------------
# Датасет с вариациями смайликов
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

        # случайный сдвиг смайлика
        offset_x = random.randint(10, 40)
        offset_y = random.randint(10, 40)

        if random.random() > 0.5:
            # Смайлик :)
            eye_y = offset_y
            eye_x_left = offset_x
            eye_x_right = offset_x + 40

            img[eye_y, eye_x_left] = 1
            img[eye_y, eye_x_right] = 1

            # Улыбка
            for i in range(eye_x_left, eye_x_right):
                j = int(offset_y + 30 + 10 * np.sin((i - eye_x_left) / 40 * np.pi))
                img[j, i] = 1

            label = 1
        else:
            # Грусть / шум
            for _ in range(50):
                x, y = random.randint(0, 99), random.randint(0, 99)
                img[y, x] = 1
            label = 0

        # добавляем случайный шум
        for _ in range(random.randint(0, 5)):
            x, y = random.randint(0, 99), random.randint(0, 99)
            img[y, x] = 1

        return img, label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# -------------------------------
# CNN модель
# -------------------------------
class SmileyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # после двух пуллингов картинка 100x100 → 25x25
        self.fc1 = nn.Linear(32 * 25 * 25, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------------
# Функции обучения
# -------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in dataloader:
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
    return avg_loss, accuracy, correct, total


def train_model(model, dataloader, max_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, max_epochs + 1):
        loss, acc, correct, total = train_one_epoch(model, dataloader, criterion, optimizer)

        print(f"Epoch {epoch:02d} | "
              f"Loss: {loss:.4f} | "
              f"Accuracy: {acc * 100:.2f}% ({correct}/{total})")

        if acc == 1.0:
            torch.save(model.state_dict(), "smiley_model_varied.pth")
            print("✅ Reached 100% accuracy! Model saved as smiley_model_varied.pth")
            break


# -------------------------------
# Запуск
# -------------------------------
if __name__ == "__main__":
    dataset = SmileyDataset(num_samples=5000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SmileyCNN().to(DEVICE)
    train_model(model, dataloader)
