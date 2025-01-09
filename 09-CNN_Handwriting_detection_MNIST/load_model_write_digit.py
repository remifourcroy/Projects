import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPainter, QImage, QPen
from PyQt5.QtCore import Qt, QPoint
from PIL import Image, ImageOps
import torch
from train import Net
import torchvision.transforms as transforms
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
logging.debug("Loading the model...")
model = Net()
model.load_state_dict(torch.load('./mnist_cnn_net.pth'))
model.eval()
logging.debug("Model loaded successfully.")

# Define transformations for input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class DrawingCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 200)
        self.setStyleSheet("border: 2px solid black;")
        self.canvas = QImage(200, 200, QImage.Format_RGB32)
        self.canvas.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            logging.debug("Mouse pressed.")
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() == Qt.LeftButton:
            logging.debug("Mouse moved.")
            painter = QPainter(self.canvas)
            pen = QPen(Qt.black, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            logging.debug("Mouse released.")
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.canvas, self.canvas.rect())

    def clearCanvas(self):
        logging.debug("Clearing canvas.")
        self.canvas.fill(Qt.white)
        self.update()

    def getCanvasImage(self):
        logging.debug("Fetching canvas image.")
        # Convert the QImage to a NumPy array
        width = self.canvas.width()
        height = self.canvas.height()
        ptr = self.canvas.bits()
        ptr.setsize(self.canvas.byteCount())
        canvas_array = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))  # RGBA format
        return canvas_array[:, :, :3]  # Extract RGB channels


class DigitRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        logging.debug("Initializing application...")
        self.setWindowTitle("Handwritten Digit Recognizer")
        self.setGeometry(100, 100, 400, 300)

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Drawing canvas
        self.canvas = DrawingCanvas()
        self.layout.addWidget(self.canvas)

        # Buttons for prediction and clearing
        button_layout = QHBoxLayout()
        self.layout.addLayout(button_layout)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_digit)
        button_layout.addWidget(self.predict_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.canvas.clearCanvas)
        button_layout.addWidget(self.clear_button)

        # Label for displaying the prediction
        self.result_label = QLabel("Predicted Digit: None")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px;")
        self.layout.addWidget(self.result_label)
        logging.debug("Application initialized successfully.")

    def predict_digit(self):
        logging.debug("Predict button clicked.")
        try:
            # Get the drawn image from the canvas
            canvas_array = self.canvas.getCanvasImage()

            # Convert to grayscale and preprocess
            pil_image = Image.fromarray(canvas_array).convert('L')  # Convert to grayscale
            pil_image = ImageOps.invert(pil_image)  # Invert colors
            bbox = pil_image.getbbox()  # Get bounding box of the digit
            if bbox:
                pil_image = pil_image.crop(bbox)  # Crop to bounding box
            pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28

            # Save for debugging
            pil_image.save("debug_input_image.png")
            logging.debug("Saved input image as debug_input_image.png")

            # Convert to tensor
            input_tensor = transform(pil_image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                output = model(input_tensor)
                logging.debug(f"Raw model output: {output}")
                _, predicted = torch.max(output.data, 1)

            logging.debug(f"Prediction complete: {predicted.item()}")
            self.result_label.setText(f"Predicted Digit: {predicted.item()}")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            self.result_label.setText("Prediction Error")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognizerApp()
    window.show()
    sys.exit(app.exec_())
