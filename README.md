# Smiley Predictor

A simple interactive Python application that lets you **draw a smiley or sad face** on a 100×100 grid and predicts what you drew using a trained **PyTorch CNN model**.

The model has been trained on **varied synthetic data**, so it can recognize faces even if drawn slightly off-center or with small variations.

---

## Features

* Draw smiley or sad faces on a 100×100 grid.
* Predict in **real-time** while drawing.
* Clear the canvas easily with:

  * **Right-click** on the canvas, or
  * **"Clear" button** under the canvas.
* Uses GPU if available for fast predictions.
* Model `smiley_model_varied.pth` is pre-trained on varied synthetic faces.

---

![Smiley Example](https://raw.githubusercontent.com/IbrokhimN/Smiley/main/sml.png)

---


## Installation

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

## Usage

1. Make sure the trained model `smiley_model_varied.pth` is in the project folder.
2. Run the GUI:

```bash
python3 draw_smiley_varied.py
```

3. **Draw** your face using the **left mouse button**.
4. **Prediction** will appear in real-time.
5. **Clear** the canvas with **right-click** or the **Clear button**.

---

## Files

* `train_smiley_varied.py` – Script to train the CNN on varied synthetic data.
* `smiley_model_varied.pth` – Pre-trained CNN model for face prediction.
* `draw_smiley_varied.py` – Interactive drawing GUI.
* `README.md` – This file.

---

## Notes

* The model is trained on synthetic data with random positions, small noise, and varied face placement.
* It works best when drawn roughly like a smiley or sad face but can handle small offsets and variations.
* For more accuracy, retrain with **more samples or different styles**.
