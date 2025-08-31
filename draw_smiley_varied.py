import tkinter as tk
import numpy as np
import torch
import torch.nn as nn

# --- Настройки ---
CELL_SIZE = 5
GRID_SIZE = 100
WINDOW_SIZE = CELL_SIZE * GRID_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# --- Загружаем модель ---
model = SmileyCNN().to(device)
model.load_state_dict(torch.load("smiley_model_varied.pth", map_location=device))
model.eval()

# --- Предсказание ---
def predict(grid):
    img_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return "Smiley" if pred==1 else "Not Smiley"

# --- GUI ---
class DrawGrid:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white")
        self.canvas.pack()
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # События
        self.canvas.bind("<B1-Motion>", self.paint_and_predict)
        self.canvas.bind("<Button-3>", self.clear)  # правая кнопка мыши

        self.label = tk.Label(master, text="Prediction: ", font=("Arial", 16))
        self.label.pack()

        self.clear_button = tk.Button(master, text="Clear", command=self.clear)
        self.clear_button.pack()

    def paint(self, event):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.grid[y, x] = 1.0
            self.canvas.create_rectangle(x*CELL_SIZE, y*CELL_SIZE, 
                                         (x+1)*CELL_SIZE, (y+1)*CELL_SIZE, 
                                         fill="black")

    def paint_and_predict(self, event):
        self.paint(event)
        result = predict(self.grid)
        self.label.config(text=f"Prediction: {result}")

    def clear(self, event=None):
        self.canvas.delete("all")
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.label.config(text="Prediction: ")

# --- Запуск GUI ---
root = tk.Tk()
root.title("Draw a Smiley or Sad Face (Varied Model)")
app = DrawGrid(root)
root.mainloop()

