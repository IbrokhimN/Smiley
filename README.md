# 😊 Smiley Predictor

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-1.13-red?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-ready-brightgreen)

**Smiley Predictor** – interactive Python application to draw a smiley 🙂 or sad 🙁 face on a `100×100` grid and get **real-time predictions** using a trained **PyTorch CNN model**.

> Works even if your drawing is slightly off-center, noisy, or has small variations.

---

## 🌟 Features

* 🎨 Draw **smiley 🙂** or **sad 🙁** faces on a **100×100 canvas**
* ⚡ **Real-time predictions** while drawing
* 🧹 **Clear canvas** easily:

  * Right-click on the canvas
  * Or press **Clear** button
* 🖌 Adjustable **brush size** & **color** (advanced GUI)
* ↩️ **Undo / Redo**
* 🖼 **Mini-map preview**
* 🚀 **GPU acceleration** supported
* 📦 Pre-trained model included: `smiley_model_varied.pth`

---

## 📸 Live Demo

![Smiley Example](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/sml.png)

> 💡 Tip: Draw roughly like a smiley or sad face. The model tolerates small offsets and noise.

---

## 🔧 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/IbrokhimN/Smiley.git
cd Smiley
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

> ✅ Recommended: Python **3.10+**

---

## ▶️ Quick Start

1. Ensure `smiley_model_varied.pth` is in the project folder
2. Launch the GUI:

```bash
python3 draw_smiley_varied.py
```

3. **Draw** using the **left mouse button**
4. **Prediction updates live**
5. **Clear** canvas: right-click or press **Clear button**
6. (Advanced) **Undo/Redo**, **change brush color** or **brush size**

---

## 📝 Notes

* Model trained on **synthetic images**:

  * Random positions
  * Small noise
  * Varied face placement
* Works best when drawing roughly resembles a **smiley or sad face**
* To improve accuracy:

  * Increase dataset size
  * Train with **different drawing styles**

---

## 🗂 Project Structure

| File                      | Description                            |
| ------------------------- | -------------------------------------- |
| `train_smiley_varied.py`  | Train the CNN on varied synthetic data |
| `smiley_model_varied.pth` | Pre-trained CNN model                  |
| `draw_smiley_varied.py`   | Interactive drawing GUI                |
| `README.md`               | This file                              |

---

## 📌 Try it Now

```bash
python3 draw_smiley_varied.py
```

> Draw a face and watch **real-time predictions**!

Optional: Embed a GIF here to show live drawing + prediction:

![GIF Example](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/sml.gif)

---

## 🔗 Resources

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [Tkinter Documentation](https://docs.python.org/3/library/tkinter.html)
* [MIT License](https://opensource.org/licenses/MIT)

---

## 📜 License

MIT License – free to use, modify, and share


