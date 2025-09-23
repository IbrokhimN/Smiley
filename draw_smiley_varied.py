import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# -----------------------------
# Настройки
# -----------------------------
CELL_SIZE = 5
GRID_SIZE = 100
WINDOW_SIZE = CELL_SIZE * GRID_SIZE
BRUSH_SIZE = 3
MAX_UNDO = 20

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
        self.master.title("Smiley Predictor Advanced GUI")

        # Canvas
        self.canvas = tk.Canvas(master, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Grid и история для undo/redo
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.history = deque(maxlen=MAX_UNDO)
        self.redo_stack = deque(maxlen=MAX_UNDO)

        # Label предсказания
        self.label = tk.Label(master, text="Prediction: ", font=("Arial", 16))
        self.label.grid(row=1, column=0, sticky="w", padx=10)

        # Кнопки управления
        button_frame = tk.Frame(master)
        button_frame.grid(row=2, column=0, pady=(5,10))
        tk.Button(button_frame, text="Clear", command=self.clear).pack(side="left", padx=5)
        tk.Button(button_frame, text="Undo", command=self.undo).pack(side="left", padx=5)
        tk.Button(button_frame, text="Redo", command=self.redo).pack(side="left", padx=5)

        # Bind события
        self.canvas.bind("<B1-Motion>", self.paint_and_predict)
        self.canvas.bind("<Button-3>", self.clear)
        self.canvas.bind("<MouseWheel>", self.change_brush_size)

    def paint(self, event: tk.Event) -> None:
        """Рисуем кистью на канвасе и обновляем сетку"""
        x_center, y_center = event.x // CELL_SIZE, event.y // CELL_SIZE
        half_brush = BRUSH_SIZE // 2
        coords = []

        for dy in range(-half_brush, half_brush + 1):
            for dx in range(-half_brush, half_brush + 1):
                x = x_center + dx
                y = y_center + dy
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    self.grid[y, x] = 1.0
                    coords.append((x, y))
                    self.canvas.create_rectangle(
                        x * CELL_SIZE, y * CELL_SIZE,
                        (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                        fill="black", outline=""
                    )
        if coords:
            self.history.append(coords)
            self.redo_stack.clear()

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
        self.history.clear()
        self.redo_stack.clear()
        self.label.config(text="Prediction: ", fg="black")

    def undo(self) -> None:
        """Отмена последнего штриха"""
        if self.history:
            last = self.history.pop()
            self.redo_stack.append(last)
            for x, y in last:
                self.grid[y, x] = 0.0
            self.redraw_canvas()
            self.update_prediction()

    def redo(self) -> None:
        """Повтор отмененного штриха"""
        if self.redo_stack:
            redo_coords = self.redo_stack.pop()
            self.history.append(redo_coords)
            for x, y in redo_coords:
                self.grid[y, x] = 1.0
            self.redraw_canvas()
            self.update_prediction()

    def redraw_canvas(self) -> None:
        """Перерисовка канваса из сетки"""
        self.canvas.delete("all")
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y, x]:
                    self.canvas.create_rectangle(
                        x * CELL_SIZE, y * CELL_SIZE,
                        (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                        fill="black", outline=""
                    )

    def update_prediction(self) -> None:
        """Обновление предсказания"""
        result = predict(self.grid)
        color = "green" if result == "Smiley" else "red"
        self.label.config(text=f"Prediction: {result}", fg=color)

    def change_brush_size(self, event: tk.Event) -> None:
        """Изменяем размер кисти колесиком мыши"""
        global BRUSH_SIZE
        if event.delta > 0 and BRUSH_SIZE < 15:
            BRUSH_SIZE += 2
        elif event.delta < 0 and BRUSH_SIZE > 1:
            BRUSH_SIZE -= 2


# -----------------------------
# Запуск
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawGrid(root)
    root.mainloop()
