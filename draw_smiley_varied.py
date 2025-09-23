import tkinter as tk
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Настройки
# -----------------------------
CELL_SIZE: int = 5
GRID_SIZE: int = 100
WINDOW_SIZE: int = CELL_SIZE * GRID_SIZE
BRUSH_SIZE: int = 3  # толщина кисти в клетках

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -----------------------------
# Модель
# -----------------------------
class SmileyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 25 * 25, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------
# Загрузка модели
# -----------------------------
model = SmileyCNN().to(DEVICE)
model.load_state_dict(torch.load("smiley_model_varied.pth", map_location=DEVICE))
model.eval()


# -----------------------------
# Предсказание
# -----------------------------
def predict(grid: np.ndarray) -> str:
    img_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return "Smiley" if pred == 1 else "Not Smiley"


# -----------------------------
# GUI
# -----------------------------
class DrawGrid:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Draw a Smiley or Sad Face (Varied Model)")

        # Canvas
        self.canvas = tk.Canvas(master, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white")
        self.canvas.pack(padx=10, pady=10)

        # Grid
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Prediction label
        self.label = tk.Label(master, text="Prediction: ", font=("Arial", 16))
        self.label.pack(pady=(0, 5))

        # Clear button
        self.clear_button = tk.Button(master, text="Clear", command=self.clear)
        self.clear_button.pack(pady=(0, 10))

        # Bind events
        self.canvas.bind("<B1-Motion>", self.paint_and_predict)
        self.canvas.bind("<Button-3>", self.clear)

    def paint(self, event: tk.Event) -> None:
        """Рисуем кистью на канвасе и обновляем сетку"""
        x_center = event.x // CELL_SIZE
        y_center = event.y // CELL_SIZE

        half_brush = BRUSH_SIZE // 2
        for dy in range(-half_brush, half_brush + 1):
            for dx in range(-half_brush, half_brush + 1):
                x = x_center + dx
                y = y_center + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    self.grid[y, x] = 1.0
                    self.canvas.create_rectangle(
                        x * CELL_SIZE, y * CELL_SIZE,
                        (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                        fill="black", outline=""
                    )

    def paint_and_predict(self, event: tk.Event) -> None:
        """Рисуем и обновляем предсказание"""
        self.paint(event)
        result = predict(self.grid)
        color = "green" if result == "Smiley" else "red"
        self.label.config(text=f"Prediction: {result}", fg=color)

    def clear(self, event: tk.Event = None) -> None:
        """Очистка канваса и сетки"""
        self.canvas.delete("all")
        self.grid.fill(0.0)
        self.label.config(text="Prediction: ", fg="black")


# -----------------------------
# Запуск
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawGrid(root)
    root.mainloop()
