
# 😊 Smiley Predictor

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-1.13-red?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-ready-brightgreen)

**Interactive Python app** to draw a smiley 🙂 or sad 🙁 face on a `100×100` grid and predict it **in real-time** using a **PyTorch CNN** model trained on **synthetic data with variations**. Works even if the drawing is slightly off-center or noisy.

---

## 🚀 Features

* 🎨 Draw **smiley 🙂** or **sad 🙁** faces on a **100×100 canvas**
* ⚡ **Real-time predictions** while drawing
* 🧹 **Clear canvas** easily:

  * **Right-click**, or
  * Press **Clear button**
* 🖌 Adjustable **brush size** and **color support** (advanced GUI)
* ↩️ **Undo / Redo** (advanced GUI)
* 🖼 **Mini-map preview** (optional)
* 🚀 **GPU acceleration** supported
* 📦 Pre-trained model: `smiley_model_varied.pth`

---

## 📸 Demo

![Smiley Example](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/sml.png)

> 💡 Tip: Draw roughly like a smiley or sad face. Model handles small offsets and noise.

---

## 🔧 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/IbrokhimN/Smiley.git
cd Smiley
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> ✅ Recommended: Python **3.10+**

---

## ▶️ Usage

1. Ensure `smiley_model_varied.pth` is in the project folder
2. Launch the GUI:

```bash
python3 draw_smiley_varied.py
```

3. **Draw** using the **left mouse button**
4. **Prediction updates live**
5. **Clear** the canvas:

   * Right-click, or
   * Press **Clear button**
6. (Advanced) **Undo/Redo**, **change brush color** or **size** if available

---

## 📝 Notes

* Model trained on **synthetic images**:

  * Random positions
  * Small noise
  * Varied face placement
* Best results when drawing roughly resembles a **smiley or sad face**
* For improved accuracy:

  * Increase dataset size
  * Train with **different drawing styles**

---

## 📌 Try it Now

```bash
python3 draw_smiley_varied.py
```

> Draw a face and watch **real-time predictions**!

---

## 📜 License

MIT License – free to use, modify, and share

