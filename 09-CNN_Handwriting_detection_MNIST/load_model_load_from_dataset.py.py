import sys
import torch
import torchvision
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
from train import Net

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the test dataset
dataset_dir = 'mnist'
testset = torchvision.datasets.MNIST(
    dataset_dir,
    train=False,
    download=False,
    transform=transform
)

# Create a DataLoader for the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

# Load the trained model
PATH = './mnist_cnn_net.pth'
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()  # Set the model to evaluation mode

# Function to convert a tensor image to a QImage
def tensor_to_qimage(tensor):
    tensor = tensor / 2 + 0.5  # Unnormalize
    npimg = tensor.numpy()
    if len(npimg.shape) == 2:  # If it's a 2D tensor, add a channel dimension
        npimg = np.expand_dims(npimg, axis=2)
    npimg = (npimg * 255).astype(np.uint8)  # Convert to 0-255 range
    height, width = npimg.shape[:2]
    qimage = QImage(npimg, width, height, QImage.Format_Grayscale8)
    return qimage


class MNISTApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MNIST Predictor")
        self.layout = QVBoxLayout()

        self.image_labels = []
        for _ in range(4):
            label = QLabel(self)
            label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(label)
            self.image_labels.append(label)

        self.prediction_label = QLabel("Predictions: ", self)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_label)

        self.reset_button = QPushButton("Reset and Predict", self)
        self.reset_button.clicked.connect(self.reset_and_predict)
        self.layout.addWidget(self.reset_button)

        self.setLayout(self.layout)
        self.reset_and_predict()

    def reset_and_predict(self):
        dataiter = iter(testloader)
        images, labels = next(dataiter)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        predictions = [str(predicted[j].item()) for j in range(4)]

        self.prediction_label.setText(f"Predictions: {' '.join(predictions)}")

        for i in range(4):
            qimage = tensor_to_qimage(images[i][0])
            pixmap = QPixmap.fromImage(qimage)
            self.image_labels[i].setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MNISTApp()
    window.show()
    sys.exit(app.exec_())
