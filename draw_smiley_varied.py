import tkinter as tk
import numpy as np
import torch
import torch.nn as nn

CELL_SIZE = 5
GRID_SIZE = 100
WINDOW_SIZE = CELL_SIZE * GRID_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
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
model = SmileyCNN().to(device)
model.load_state_dict(torch.load("smiley_model_varied.pth", map_location=device))
model.eval()

def predict(grid):
    img_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return "Smiley 😊" if pred==1 else "Not Smiley 😢"

class DrawGrid:
    def __init__(self, master):
        self.master = master
        self.master.configure(bg="#f0f0f0")
        self.canvas = tk.Canvas(master, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white", highlightthickness=2, highlightbackground="black")
        self.canvas.pack(padx=10, pady=10)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.canvas.bind("<B1-Motion>", self.paint_and_predict)
        self.canvas.bind("<Button-1>", self.paint_and_predict) 
        self.canvas.bind("<Button-3>", self.clear)  
        self.label = tk.Label(master, text="Prediction: ", font=("Arial", 16), bg="#f0f0f0")
        self.label.pack(pady=5)
        self.button_frame = tk.Frame(master, bg="#f0f0f0")
        self.button_frame.pack(pady=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear, bg="#ffcccc", fg="black", font=("Arial", 12), width=10)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.exit_button = tk.Button(self.button_frame, text="Exit", command=master.quit, bg="#ccccff", fg="black", font=("Arial", 12), width=10)
        self.exit_button.pack(side=tk.LEFT, padx=5)

    def paint(self, event):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.grid[y, x] = 1.0
            self.canvas.create_rectangle(x*CELL_SIZE, y*CELL_SIZE, 
                                         (x+1)*CELL_SIZE, (y+1)*CELL_SIZE, 
                                         fill="black", outline="#888888")

    def paint_and_predict(self, event):
        self.paint(event)
        result = predict(self.grid)
        self.label.config(text=f"Prediction: {result}")

    def clear(self, event=None):
        self.canvas.delete("all")
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.label.config(text="Prediction: ")
root = tk.Tk()
root.title("Draw a Smiley or Sad Face 😀")
root.resizable(False, False)
app = DrawGrid(root)
root.mainloop()
