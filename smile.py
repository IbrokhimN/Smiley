import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Dataset с вариациями ---
class SmileyDataset(Dataset):
    def __init__(self, num_samples=5000):
        self.num_samples = num_samples
        data_list = []
        labels_list = []
        for _ in range(num_samples):
            img, label = self.generate_sample()
            data_list.append(img)
            labels_list.append(label
        self.data = torch.tensor(np.array(data_list), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(np.array(labels_list), dtype=torch.long)

    def generate_sample(self):
        img = np.zeros((100, 100), dtype=np.float32)

        # случайный сдвиг смайлика
        offset_x = random.randint(10, 40)
        offset_y = random.randint(10, 40)

        if random.random() > 0.5:
            # Smiley
            eye_y = offset_y
            eye_x_left = offset_x
            eye_x_right = offset_x + 40

            img[eye_y, eye_x_left] = 1
            img[eye_y, eye_x_right] = 1

            # улыбка
            for i in range(eye_x_left, eye_x_right):
                j = int(offset_y + 30 + 10 * np.sin((i-eye_x_left)/40*np.pi))
                img[j, i] = 1

            label = 1
        else:
            # Sad / random noise
            for _ in range(50):
                x = random.randint(0, 99)
                y = random.randint(0, 99)
                img[y, x] = 1
            label = 0

        # можно добавить маленький шум
        for _ in range(random.randint(0,5)):
            x = random.randint(0, 99)
            y = random.randint(0, 99)
            img[y, x] = 1

        return img, label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Модель ---
class SmileyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*25*25, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Обучение ---
dataset = SmileyDataset(num_samples=5000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SmileyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch = 0
while True:
    epoch += 1
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch} | Loss: {running_loss/len(dataloader):.4f} | Accuracy: {accuracy*100:.2f}% | Correct: {correct}/{total}")

    # Если достигнута 100% точность, сохраняем и выходим
    if accuracy == 1.0:
        torch.save(model.state_dict(), "smiley_model_varied.pth")
        print("Reached 100% accuracy! Model saved as smiley_model_varied.pth")
        break

