# 😊 Smiley Predictor – Pro Edition

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-1.13-red?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-ready-brightgreen)

**Smiley Predictor** – interactive Python app to draw a smiley 🙂 or sad 🙁 face on a `100×100` grid and get **real-time predictions** using a **PyTorch CNN** trained on **synthetic varied data**.

> Works even if the drawing is slightly off-center or noisy.

---

## 🌟 Features

* 🎨 Draw smiley 🙂 or sad 🙁 faces on a **100×100 canvas**
* ⚡ **Real-time predictions**
* 🧹 **Clear canvas** (Right-click or **Clear button**)
* 🖌 Adjustable **brush size** & **color**
* ↩️ **Undo / Redo** (advanced GUI)
* 🚀 **GPU acceleration**
* 📦 Pre-trained model: `smiley_model_varied.pth`

---

## 🏗 Model Architecture

```text
Input: 1 × 100 × 100
 └─ Conv2d(1 → 16, 3x3, padding=1) → ReLU
     └─ Conv2d(16 → 32, 3x3, padding=1) → ReLU
         └─ MaxPool2d(2x2)
             └─ MaxPool2d(2x2)
                 └─ Flatten
                     └─ Linear(32*25*25 → 64) → ReLU
                         └─ Linear(64 → 2) → Softmax
Output: Smiley / Not Smiley
```

> This simple CNN handles **shifted, noisy, or varied faces**.

---

## 📈 Training Metrics (Example)

<div align="center">

![Loss vs Epochs](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/loss_plot.png)
*Training & Validation Loss*

![Accuracy vs Epochs](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/accuracy_plot.png)
*Training & Validation Accuracy*

</div>

> ✅ Reached **\~100% accuracy** on synthetic varied data.

---

## 📸 Prediction Examples

<div align="center">

| Drawing                                                                     | Prediction        |
| --------------------------------------------------------------------------- | ----------------- |
| ![Smiley](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/sml.png)  | **Smiley 🙂**     |
| ![Noise](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/noise.png) | **Not Smiley 🙁** |

</div>

---

## 🔧 Installation

```bash
git clone https://github.com/IbrokhimN/Smiley.git
cd Smiley
pip install -r requirements.txt
```

> Python **3.10+** recommended

---

## ▶️ Usage

```bash
python3 draw_smiley_varied.py
```

* Draw face using **left mouse button**
* Prediction updates **in real-time**
* Clear canvas: **Right-click** or **Clear button**

---

## 📝 Notes

* Model trained on **synthetic images** with:

  * Random positions
  * Small noise
  * Varied face placement
* Works best with rough **smiley/sad shapes**
* For better accuracy:

  * Increase dataset size
  * Include **different drawing styles**

---

## 🗂 Project Structure

| File                      | Description                        |
| ------------------------- | ---------------------------------- |
| `train_smiley_varied.py`  | Train CNN on synthetic varied data |
| `smiley_model_varied.pth` | Pre-trained CNN model              |
| `draw_smiley_varied.py`   | Interactive GUI                    |
| `README.md`               | This file                          |

---

## 📌 Try it Now

```bash
python3 draw_smiley_varied.py
```

> Draw a face and watch **real-time predictions**!

---

## 📜 License

MIT License – free to use, modify, and share

