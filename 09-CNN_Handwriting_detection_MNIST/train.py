import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import torch.nn.functional as F #

''' For any contact : remifourcroy.pro@gmail.com'''


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Specify the dataset directory
dataset_dir = 'mnist'

# Check if the dataset directory exists
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print('Loading training dataset...')
# Load the MNIST dataset without re-downloading if it already exists
trainset = torchvision.datasets.MNIST(
    dataset_dir,
    train=True,
    download=not os.path.exists(os.path.join(dataset_dir, 'MNIST/raw')),
    transform=transform
)
print('Loading testing dataset...')
testset = torchvision.datasets.MNIST(
    dataset_dir,
    train=False,
    download=not os.path.exists(os.path.join(dataset_dir, 'MNIST/raw')),
    transform=transform
)
print("Number of image samples for training =",trainset.data.shape[0])
print("Number of image samples for testing =",testset.data.shape[0])

trainloader = torch.utils.data.DataLoader(trainset,
                                           batch_size = 128,
                                           shuffle = True,
                                           num_workers = 0)

testloader = torch.utils.data.DataLoader(testset,
                                          batch_size = 128,
                                          shuffle = False,
                                          num_workers = 0)
class Net(nn.Module):
    def __init__(self):
        # super is a subclass of the nn.Module and inherits all its methods
        super(Net, self).__init__()

        # We define our layer objects here
        # Our first CNN Layer using 32 Fitlers of 3x3 size, with stride of 1 & padding of 0
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Our second CNN Layer using 64 Fitlers of 3x3 size, with stride of 1 & padding of 0
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Our Max Pool Layer 2 x 2 kernel of stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Our first Fully Connected Layer (called Linear), takes the output of our Max Pool
        # which is 12 x 12 x 64 and connects it to a set of 128 nodes
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # Our second Fully Connected Layer, connects the 128 nodes to 10 output nodes (our classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # here we define our forward propogation sequence
        # Remember it's Conv1 - Relu - Conv2 - Relu - Max Pool - Flatten - FC1 - FC2
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model and move it (memory and operations) to the CUDA device
net = Net()
net.to('cpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 10

# Create some empty arrays to store logs
epoch_log = []
loss_log = []
accuracy_log = []


if __name__ == "__main__":
    # Iterate for a specified number of epochs
    for epoch in range(epochs):
        print(f'Starting Epoch: {epoch + 1}...')

        # We keep adding or accumulating our loss after each mini-batch in running_loss
        running_loss = 0.0

        # We iterate through our trainloader iterator
        # Each cycle is a minibatch
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Move our data to GPU
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')

            # Clear the gradients before training by setting to zero
            # Required for a fresh start
            optimizer.zero_grad()

            # Forward -> backprop + optimize
            outputs = net(inputs)  # Forward Propagation
            loss = criterion(outputs, labels)  # Get Loss (quantify the difference between the results and predictions)
            loss.backward()  # Back propagate to obtain the new gradients for all nodes
            optimizer.step()  # Update the gradients/weights

            # Print Training statistics - Epoch/Iterations/Loss/Accuracy
            running_loss += loss.item()
            if i % 50 == 49:  # show our loss every 50 mini-batches
                correct = 0  # Initialize our variable to hold the count for the correct predictions
                total = 0  # Initialize our variable to hold the count of the number of labels iterated

                # We don't need gradients for validation, so wrap in
                # no_grad to save memory
                with torch.no_grad():
                    # Iterate through the testloader iterator
                    for data in testloader:
                        images, labels = data
                        # Move our data to GPU
                        images = images.to('cpu')
                        labels = labels.to('cpu')

                        # Foward propagate our test data batch through our model
                        outputs = net(images)

                        # Get predictions from the maximum value of the predicted output tensor
                        # we set dim = 1 as it specifies the number of dimensions to reduce
                        _, predicted = torch.max(outputs.data, dim=1)
                        # Keep adding the label size or length to the total variable
                        total += labels.size(0)
                        # Keep a running total of the number of predictions predicted correctly
                        correct += (predicted == labels).sum().item()

                    accuracy = 100 * correct / total
                    epoch_num = epoch + 1
                    actual_loss = running_loss / 50
                    print(
                        f'Epoch: {epoch_num}, Mini-Batches Completed: {(i + 1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                    running_loss = 0.0

        # Store training stats after each epoch
        epoch_log.append(epoch_num)
        loss_log.append(actual_loss)
        accuracy_log.append(accuracy)

    print(epoch_log, loss_log, accuracy_log)
    print('Finished Training')
    PATH = './mnist_cnn_net.pth'
    torch.save(net.state_dict(), PATH)
    print('Model saved to', PATH)

