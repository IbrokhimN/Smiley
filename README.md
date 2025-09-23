# 😊 Smiley Predictor

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-1.13-red?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-ready-brightgreen)

**Interactive Python app** to draw a smiley 🙂 or sad 🙁 face on a `100×100` grid and predict it **in real-time** using a trained **PyTorch CNN model**.

The model is trained on **synthetic data with variations**, so it works even if the drawing is slightly off-center or noisy.

---

## ✨ Features

* 🎨 Draw smiley 🙂 or sad 🙁 faces on a **100×100 canvas**.
* ⚡ **Real-time prediction** while drawing.
* 🧹 Clear the canvas easily:

  * **Right-click**, or
  * Press **Clear button**.
* 🚀 **GPU acceleration** supported.
* 📦 Pre-trained model included: `smiley_model_varied.pth`.

---

## 📸 Example

![Smiley Example](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/sml.png)

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

---

## ▶️ Usage

1. Make sure the **pre-trained model** is in the project folder: `smiley_model_varied.pth`.
2. Launch the interactive GUI:

```bash
python3 draw_smiley_varied.py
```

3. **Draw your face** with the **left mouse button**.
4. The **prediction updates in real-time**.
5. Reset the canvas: **right-click** or press **Clear**.

---

## 📝 Notes

* Trained on **synthetic images** with:

  * Random positions
  * Small noise
  * Varied face placement
* Works best when the drawing roughly resembles a **smiley or sad face**.
* For improved accuracy:

  * Increase dataset size
  * Train with **different drawing styles**

---

## 📜 License

MIT License – free to use, modify, and share.
