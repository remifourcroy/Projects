# Custom Sobel Filter Implementation

This repository contains an implementation of the **Sobel filter**, a popular edge-detection algorithm used in image processing. The Sobel filter computes the gradient magnitude in both horizontal and vertical directions to identify edges in images.

We have implemented two versions:
1. **Using `scipy.signal.convolve2d`**: A concise and efficient implementation leveraging SciPy's optimized convolution function.
2. **Without external libraries for convolution**: A detailed, manual implementation using nested loops to compute the convolution operation step by step.

---

## Features
- Apply the Sobel filter to grayscale or color images.
- Compute edge detection with horizontal and vertical gradients.
- Two implementations available:
  - **Efficient implementation**: Uses `convolve2d` from SciPy for convolution.
  - **Manual implementation**: Performs convolution using nested loops for learning purposes.
- Handles edge pixels using **padding techniques** (e.g., reflective padding).

---

## Requirements
- Python 3.x
- Required libraries:
  - `numpy`
  - `scipy`
  - `opencv-python`
  - `matplotlib` (for visualization)

Install the required libraries with:

```bash
pip install numpy scipy opencv-python matplotlib
