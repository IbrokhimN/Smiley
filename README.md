# 😊 Smiley Predictor

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-1.13-red?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-ready-brightgreen)

**Interactive Python app** to draw a smiley 🙂 or sad 🙁 face on a `100×100` grid and predict it **in real-time** using a trained **PyTorch CNN model**.

The model is trained on **synthetic data with variations**, so it works even if the drawing is slightly off-center, noisy, or small variations are present.

---

## ✨ Features

* 🎨 Draw smiley 🙂 or sad 🙁 faces on a **100×100 canvas**.
* ⚡ **Real-time prediction** as you draw.
* 🧹 **Clear canvas** easily:

  * **Right-click** on canvas
  * Or press the **Clear button**
* 🖌 Adjustable brush size and color support (if using enhanced version)
* 🚀 **GPU acceleration** supported for fast predictions
* 📦 **Pre-trained model** included: `smiley_model_varied.pth`
* 🖼 Preview mini-map (optional in advanced GUI)
* ↩️ **Undo / Redo** functionality (advanced GUI)

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

> 💡 Recommended Python version: **3.10+**

---

## ▶️ Usage

1. Ensure the pre-trained model is present: `smiley_model_varied.pth`
2. Launch the interactive GUI:

```bash
python3 draw_smiley_varied.py
```

3. **Draw** your face with the **left mouse button**.
4. The **prediction updates in real-time**.
5. **Clear** the canvas:

   * Right-click, or
   * Press the **Clear button**
6. (Advanced) Use **Undo / Redo** and **change brush size or color** if enabled.

---

## 📝 Notes

* Model trained on **synthetic images** with:

  * Random positions
  * Small noise
  * Varied face placement
* Works best when drawing roughly resembles a **smiley or sad face**.
* To improve accuracy:

  * Increase dataset size
  * Train with **different drawing styles**

---

## 🗂 Files Overview

| File                      | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `train_smiley_varied.py`  | Script to train the CNN on varied synthetic data |
| `smiley_model_varied.pth` | Pre-trained CNN model for face prediction        |
| `draw_smiley_varied.py`   | Interactive drawing GUI                          |
| `README.md`               | This file                                        |

---

## 📜 License

MIT License – free to use, modify, and share.

