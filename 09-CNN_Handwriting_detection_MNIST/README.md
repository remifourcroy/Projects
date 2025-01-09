# Handwritten Digit Recognition (PyTorch)

This project demonstrates a handwritten digit recognition system using PyTorch, trained on the MNIST dataset. The project includes:

1. **Training a CNN model (`train.py`)**
2. **Visualizing predictions (`load_model_load_from_dataset.py`)**
3. **Interactive digit prediction using a drawing canvas (`load_model_write_digit.py`)**

---

## Features

- **Model Training**: A convolutional neural network (CNN) trained on the MNIST dataset.
- **Batch Predictions**: Display and predict digits for random test images.
- **Interactive Drawing Canvas**: Draw digits and get predictions in real-time.

---

## Requirements

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

**Libraries:**
- `torch`
- `torchvision`
- `PyQt5`
- `numpy`
- `Pillow`

---

## Files Overview

1. **`train.py`**
   - Trains a CNN model on the MNIST dataset.
   - Saves the trained model to `mnist_cnn_net.pth`.

2. **`load_model_load_from_dataset.py`**
   - Loads the trained model.
   - Displays 4 random images from the MNIST test dataset with their predicted labels.

3. **`load_model_write_digit.py`**
   - Provides an interactive GUI for handwritten digit recognition.
   - Allows users to draw a digit and see the model's prediction.

---

## How to Use

### 1. Train the Model

Run the `train.py` script to train the CNN model:

```bash
python train.py
```

This will save the trained model as `mnist_cnn_net.pth`.

### 2. Batch Predictions

Run `load_model_load_from_dataset.py` to display and predict digits from the MNIST test dataset:

```bash
python load_model_load_from_dataset.py
```

- The UI displays 4 random images and their predicted labels.
- Use the "Reset and Predict" button to load a new set of random images.

### 3. Interactive Predictions

Run `load_model_write_digit.py` to use the interactive drawing canvas:

```bash
python load_model_write_digit.py
```

- Draw a digit on the canvas.
- Click "Predict" to see the predicted digit.
- Click "Clear" to reset the canvas.

---

## Directory Structure

```
.
├── train.py
├── load_model_load_from_dataset.py
├── load_model_write_digit.py
├── mnist_cnn_net.pth
├── requirements.txt
├── README.md
└── mnist/  # Directory for the MNIST dataset
```

---

## Notes

1. Ensure the `mnist_cnn_net.pth` file is in the same directory as the scripts.
2. The MNIST dataset will be automatically downloaded when running `train.py` for the first time.
3. For troubleshooting, refer to the logs or check if all libraries are installed.

---

## Contact

For any questions or issues, contact:
**remifourcroy.pro@gmail.com**
