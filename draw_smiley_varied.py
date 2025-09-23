import tkinter as tk
from tkinter import filedialog, colorchooser
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from PIL import Image

# -----------------------------
# Настройки
# -----------------------------
CELL_SIZE = 5
GRID_SIZE = 100
WINDOW_SIZE = CELL_SIZE * GRID_SIZE
BRUSH_SIZE = 3
MAX_UNDO = 50
PREVIEW_SCALE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# Модель
# -----------------------------
class SmileyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*25*25,64)
        self.fc2 = nn.Linear(64,2)
    def forward(self,x):
        x=torch.relu(self.conv1(x))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# Загрузка модели
# -----------------------------
model=SmileyCNN().to(DEVICE)
model.load_state_dict(torch.load("smiley_model_varied.pth", map_location=DEVICE))
model.eval()

# -----------------------------
# Предсказание
# -----------------------------
def predict_prob(grid: np.ndarray):
    img_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs[1], probs[0]

# -----------------------------
# GUI
# -----------------------------
class SmileyEditor:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Smiley Predictor Pro")
        
        # Canvas
        self.canvas = tk.Canvas(master,width=WINDOW_SIZE,height=WINDOW_SIZE,bg="white")
        self.canvas.grid(row=0,column=0,padx=10,pady=10)
        
        self.preview = tk.Canvas(master,width=GRID_SIZE//PREVIEW_SCALE,height=GRID_SIZE//PREVIEW_SCALE,bg="white")
        self.preview.grid(row=0,column=1,padx=10)
        
        # Grid и история
        self.grid = np.zeros((GRID_SIZE,GRID_SIZE),dtype=np.float32)
        self.history = deque(maxlen=MAX_UNDO)
        self.redo_stack = deque(maxlen=MAX_UNDO)
        
        # Prediction
        self.label = tk.Label(master,text="Smiley: 0.00 | Not Smiley: 0.00",font=("Arial",14))
        self.label.grid(row=1,column=0,columnspan=2,sticky="w",padx=10)
        
        # Кнопки управления
        btn_frame=tk.Frame(master)
        btn_frame.grid(row=2,column=0,columnspan=2,pady=(5,10))
        tk.Button(btn_frame,text="Clear",command=self.clear).pack(side="left",padx=5)
        tk.Button(btn_frame,text="Undo",command=self.undo).pack(side="left",padx=5)
        tk.Button(btn_frame,text="Redo",command=self.redo).pack(side="left",padx=5)
        tk.Button(btn_frame,text="Save",command=self.save_image).pack(side="left",padx=5)
        tk.Button(btn_frame,text="Brush Color",command=self.choose_color).pack(side="left",padx=5)
        tk.Button(btn_frame,text="Background Color",command=self.choose_bg).pack(side="left",padx=5)
        tk.Button(btn_frame,text="Zoom In",command=lambda:self.change_zoom(2)).pack(side="left",padx=5)
        tk.Button(btn_frame,text="Zoom Out",command=lambda:self.change_zoom(0.5)).pack(side="left",padx=5)
        
        # Bind events
        self.canvas.bind("<B1-Motion>",self.paint_and_update)
        self.canvas.bind("<MouseWheel>",self.change_brush)
        self.color="black"
        self.bg="white"
        self.zoom=1.0

    def paint(self,event):
        global CELL_SIZE
        x_center=int(event.x/(CELL_SIZE*self.zoom))
        y_center=int(event.y/(CELL_SIZE*self.zoom))
        half=BRUSH_SIZE//2
        coords=[]
        for dy in range(-half,half+1):
            for dx in range(-half,half+1):
                x,y=x_center+dx,y_center+dy
                if 0<=x<GRID_SIZE and 0<=y<GRID_SIZE:
                    self.grid[y,x]=1.0
                    coords.append((x,y))
                    self.canvas.create_rectangle(
                        x*CELL_SIZE*self.zoom,y*CELL_SIZE*self.zoom,
                        (x+1)*CELL_SIZE*self.zoom,(y+1)*CELL_SIZE*self.zoom,
                        fill=self.color,outline=""
                    )
        if coords:
            self.history.append(coords)
            self.redo_stack.clear()
            self.update_preview(coords)
    
    def paint_and_update(self,event):
        self.paint(event)
        self.update_prediction()
    
    def update_prediction(self):
        smiley_prob,not_prob=predict_prob(self.grid)
        self.label.config(text=f"Smiley: {smiley_prob:.2f} | Not Smiley: {not_prob:.2f}",
                          fg=self.get_color(smiley_prob))
    
    def get_color(self,smiley_prob:float)->str:
        r=int((1-smiley_prob)*255)
        g=int(smiley_prob*255)
        return f"#{r:02x}{g:02x}00"
    
    def clear(self,event=None):
        self.canvas.delete("all")
        self.preview.delete("all")
        self.grid.fill(0.0)
        self.history.clear()
        self.redo_stack.clear()
        self.label.config(text="Smiley: 0.00 | Not Smiley: 0.00",fg="black")
    
    def undo(self):
        if self.history:
            last=self.history.pop()
            self.redo_stack.append(last)
            for x,y in last:
                self.grid[y,x]=0.0
            self.redraw()
            self.update_prediction()
    
    def redo(self):
        if self.redo_stack:
            redo_coords=self.redo_stack.pop()
            self.history.append(redo_coords)
            for x,y in redo_coords:
                self.grid[y,x]=1.0
            self.redraw()
            self.update_prediction()
    
    def redraw(self):
        self.canvas.delete("all")
        self.preview.delete("all")
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y,x]:
                    self.canvas.create_rectangle(
                        x*CELL_SIZE*self.zoom,y*CELL_SIZE*self.zoom,
                        (x+1)*CELL_SIZE*self.zoom,(y+1)*CELL_SIZE*self.zoom,
                        fill=self.color,outline=""
                    )
                    self.preview.create_rectangle(
                        x//PREVIEW_SCALE,y//PREVIEW_SCALE,
                        x//PREVIEW_SCALE+1,y//PREVIEW_SCALE+1,
                        fill=self.color,outline=""
                    )
    
    def update_preview(self,coords):
        for x,y in coords:
            self.preview.create_rectangle(
                x//PREVIEW_SCALE,y//PREVIEW_SCALE,
                x//PREVIEW_SCALE+1,y//PREVIEW_SCALE+1,
                fill=self.color,outline=""
            )
    
    def change_brush(self,event):
        global BRUSH_SIZE
        if event.delta>0 and BRUSH_SIZE<15:
            BRUSH_SIZE+=1
        elif event.delta<0 and BRUSH_SIZE>1:
            BRUSH_SIZE-=1
    
    def choose_color(self):
        c=colorchooser.askcolor(title="Choose brush color")[1]
        if c:self.color=c
    
    def choose_bg(self):
        c=colorchooser.askcolor(title="Choose background color")[1]
        if c:
            self.bg=c
            self.canvas.config(bg=self.bg)
            self.preview.config(bg=self.bg)
    
    def save_image(self):
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG files","*.png")])
        if path:
            img=(self.grid*255).astype(np.uint8)
            im=Image.fromarray(img)
            im=im.resize((GRID_SIZE*CELL_SIZE,GRID_SIZE*CELL_SIZE),Image.NEAREST)
            im.save(path)

    def change_zoom(self,factor:float):
        self.zoom*=factor
        self.redraw()

# -----------------------------
# Запуск
# -----------------------------
if __name__=="__main__":
    root=tk.Tk()
    app=SmileyEditor(root)
    root.mainloop()
