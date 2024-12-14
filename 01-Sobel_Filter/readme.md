# Custom Sobel Filter Implementation

This is an implementation of the **Sobel filter**, a popular edge-detection algorithm used in image processing. The Sobel filter computes the gradient magnitude in both horizontal and vertical directions to identify edges in images.

I have implemented two versions:
1. **Using `scipy.signal.convolve2d`**: A concise and efficient implementation leveraging SciPy's optimized convolution function.
2. **Without libraries **: Manual implementation using nested loops to compute the convolution operation step by step.

---

## Requirements
- Python 3.10.13
- Required libraries:
  - `numpy`
  - `scipy`
  - `opencv-python`
  - `matplotlib` (for visualization)
