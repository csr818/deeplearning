"""
A program to realize a simple fully connected neural network
using Pytorch library. Refer to Aladdin's codes on YouTube.
The website link is https://www.youtube.com/watch?v=Jy4wM2X21u0.

Programmed by csr.
* 2024-01-11: Initial version

Try different optimizer, finds that Adam performs better than SGD
Adam accuracy: around 97%
SGD accuracy: around 85%
"""
# Imports
import torch
import torch.nn as nn  # inner network modules and loss functions
import torch.optim as optim  # optimization like stochastic gradient descent
import torch.nn.functional as F  # activation like relu, tanh, sigmoid
from torch.utils.data import DataLoader  # help to create mini batch data to train
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


# create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Here we define the layers of the fully connected network, we use three linear layer to
        represent the first input_size to hidden_size and then fully connected to the hidden_size layer,
        at last output layer.
        :param input_size: size of the input, in this case 784(28x28)
        :param hidden_size: size of the hidden, in this case 16
        :param num_classes: size of the output, in this case 10
        """
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Set device cuda for GPU if it's available otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
hidden_size = 50
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load Data

# transforms.ToTensor() is a image transformed operation, transform PIL image to a tensor vector
train_dataset = datasets.MNIST(root='dataset/', train=True,
                               download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor())
# create a loader which is responsible for loading dataset into the model in batches
# parameter shuffle means whether shuffle randomly(True) or not (False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer

# model.parameters() returns a iteration of the learnable parameters in model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
for epoch in range(num_epochs):
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device=device), target.to(device=device)

        # transform data to the correct shape
        # -1 means flatten
        data = data.reshape(data.shape[0], -1)

        # forward
        # parameter data will be transfer to forward method realized in the class NN
        # scores is batch_size X num_classes
        scores = model(data)
        # criterion is CrossEntropy function, including a softmax operation
        # no need to find the max value index by yourself
        loss = criterion(scores, target)

        # backward
        # clear the gradients before backward
        optimizer.zero_grad()
        # calculate the gradients of weights and biases
        loss.backward()
        # update model parameters
        optimizer.step()


# check accuracy on train&test dataset, see how good the model is
def check_accuracy(loader, model):
    # check the loader.dataset is train or test
    if loader.dataset.train:
        print("Checking accuracy on train data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    # change model mode to evaluate
    model.eval()

    # avoid wasted gradient calculation
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device=device), y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            # scores are batch_size x num_classes, calculate a result of a whole batch one time
            scores = model(x)
            # 1 presents the dimension, the first return value is value, second is index
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    print(f"Got {num_correct}/{num_samples} with accuracy {100.0 * num_correct / num_samples : .2f}")
    model.train()
    return num_correct / num_samples


# f-strings to embed expressions in strings using curly braces and use : specify formatting options
print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}")
# save the parameter state directory
torch.save(model.state_dict(), 'save/fullyNet_model.pth')
