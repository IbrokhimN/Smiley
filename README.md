# 😊 Smiley Predictor

An interactive Python app that lets you **draw a smiley or sad face** on a `100×100` grid and instantly predicts what you drew using a trained **PyTorch CNN model**.

The model has been trained on **synthetic data with variations** (offsets, noise, different positions), so it can recognize faces even when they are not perfectly centered or drawn exactly the same.

---

## ✨ Features

* 🎨 Draw smiley 🙂 or sad 🙁 faces on a **100×100 grid**.
* ⚡ **Real-time prediction** while drawing.
* 🧹 Easy canvas reset:

  * **Right-click** on the canvas, or
  * **Press "Clear" button**.
* 🚀 GPU acceleration supported if available.
* 📦 Pre-trained model `smiley_model_varied.pth` included.

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

1. Ensure the trained model `smiley_model_varied.pth` is present in the project folder.
2. Run the GUI:

```bash
python3 draw_smiley_varied.py
```

3. **Draw your face** with the **left mouse button**.
4. Prediction will appear **instantly** below the canvas.
5. To reset the canvas: **right-click** or press **Clear**.

---

## 📂 Project Structure

* `train_smiley_varied.py` → Train the CNN on synthetic smiley/sad data.
* `smiley_model_varied.pth` → Pre-trained CNN weights.
* `draw_smiley_varied.py` → Interactive drawing + prediction GUI.
* `README.md` → Project documentation (this file).

---

## 📝 Notes

* The model was trained on **synthetic images** with:

  * Random positions
  * Noise injection
  * Variation in shape placement

* Works best when the face resembles a simple **smiley or sad face**.

* For higher accuracy:

  * Retrain with **larger datasets**
  * Add **more styles of drawings**

---

## 📜 License

MIT License – feel free to use, modify, and share.
