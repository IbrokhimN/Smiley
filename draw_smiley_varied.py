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
PREVIEW_SCALE = 4  # масштаб мини-карты

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
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*25*25,64)
        self.fc2 = nn.Linear(64,2)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0),-1)
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
def predict_prob(grid: np.ndarray):
    """Возвращает вероятность Smiley и Not Smiley"""
    img_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs[1], probs[0]  # Smiley, Not Smiley

# -----------------------------
# GUI
# -----------------------------
class DrawGrid:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Smiley Predictor Advanced GUI")
        
        # Canvas основной
        self.canvas = tk.Canvas(master, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white")
        self.canvas.grid(row=0,column=0,padx=10,pady=10)

        # Canvas мини-карты
        self.preview = tk.Canvas(master, width=GRID_SIZE//PREVIEW_SCALE, height=GRID_SIZE//PREVIEW_SCALE, bg="white")
        self.preview.grid(row=0,column=1,padx=10)

        # Сетка и история
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.history = deque(maxlen=MAX_UNDO)
        self.redo_stack = deque(maxlen=MAX_UNDO)

        # Prediction label
        self.label = tk.Label(master, text="Smiley: 0.00 | Not Smiley: 0.00", font=("Arial",14))
        self.label.grid(row=1,column=0,columnspan=2, sticky="w", padx=10)

        # Кнопки управления
        button_frame = tk.Frame(master)
        button_frame.grid(row=2,column=0,columnspan=2,pady=(5,10))
        tk.Button(button_frame,text="Clear",command=self.clear).pack(side="left", padx=5)
        tk.Button(button_frame,text="Undo",command=self.undo).pack(side="left", padx=5)
        tk.Button(button_frame,text="Redo",command=self.redo).pack(side="left", padx=5)

        # Bind events
        self.canvas.bind("<B1-Motion>", self.paint_and_update)
        self.canvas.bind("<Button-3>", self.clear)
        self.canvas.bind("<MouseWheel>", self.change_brush)

    def paint(self,event:tk.Event)->None:
        x_center, y_center = event.x//CELL_SIZE, event.y//CELL_SIZE
        half = BRUSH_SIZE//2
        coords=[]
        for dy in range(-half, half+1):
            for dx in range(-half, half+1):
                x, y = x_center+dx, y_center+dy
                if 0<=x<GRID_SIZE and 0<=y<GRID_SIZE:
                    self.grid[y,x]=1.0
                    coords.append((x,y))
                    self.canvas.create_rectangle(
                        x*CELL_SIZE, y*CELL_SIZE,
                        (x+1)*CELL_SIZE, (y+1)*CELL_SIZE,
                        fill="black", outline=""
                    )
        if coords:
            self.history.append(coords)
            self.redo_stack.clear()
            self.update_preview(coords)

    def paint_and_update(self,event:tk.Event)->None:
        self.paint(event)
        self.update_prediction()

    def update_prediction(self)->None:
        smiley_prob, not_prob = predict_prob(self.grid)
        self.label.config(text=f"Smiley: {smiley_prob:.2f} | Not Smiley: {not_prob:.2f}",
                          fg=self.get_color(smiley_prob))
    
    def get_color(self,smiley_prob:float)->str:
        r = int((1-smiley_prob)*255)
        g = int(smiley_prob*255)
        return f"#{r:02x}{g:02x}00"

    def clear(self,event:tk.Event=None)->None:
        self.canvas.delete("all")
        self.preview.delete("all")
        self.grid.fill(0.0)
        self.history.clear()
        self.redo_stack.clear()
        self.label.config(text="Smiley: 0.00 | Not Smiley: 0.00", fg="black")

    def undo(self)->None:
        if self.history:
            last = self.history.pop()
            self.redo_stack.append(last)
            for x,y in last:
                self.grid[y,x]=0.0
            self.redraw()
            self.update_prediction()

    def redo(self)->None:
        if self.redo_stack:
            redo_coords = self.redo_stack.pop()
            self.history.append(redo_coords)
            for x,y in redo_coords:
                self.grid[y,x]=1.0
            self.redraw()
            self.update_prediction()

    def redraw(self)->None:
        self.canvas.delete("all")
        self.preview.delete("all")
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y,x]:
                    self.canvas.create_rectangle(
                        x*CELL_SIZE, y*CELL_SIZE,
                        (x+1)*CELL_SIZE, (y+1)*CELL_SIZE,
                        fill="black", outline=""
                    )
                    self.preview.create_rectangle(
                        x//PREVIEW_SCALE, y//PREVIEW_SCALE,
                        x//PREVIEW_SCALE+1, y//PREVIEW_SCALE+1,
                        fill="black", outline=""
                    )

    def update_preview(self,coords):
        for x,y in coords:
            self.preview.create_rectangle(
                x//PREVIEW_SCALE, y//PREVIEW_SCALE,
                x//PREVIEW_SCALE+1, y//PREVIEW_SCALE+1,
                fill="black", outline=""
            )

    def change_brush(self,event:tk.Event)->None:
        global BRUSH_SIZE
        if event.delta>0 and BRUSH_SIZE<15:
            BRUSH_SIZE+=2
        elif event.delta<0 and BRUSH_SIZE>1:
            BRUSH_SIZE-=2

# -----------------------------
# Запуск
# -----------------------------
if __name__=="__main__":
    root=tk.Tk()
    app=DrawGrid(root)
    root.mainloop()
