"""
A program to realize a simple CNN using pytorch library.
Refer to Aladdin's codes on YouTube.
The website link is https://www.youtube.com/watch?v=wnK3uWv_WkU.

Programmed by csr.
* 2024-01-14: initial version

----------------------------------------------------------------
check accuracy on train dataset
Got 58943/60000 with accuracy  98.24
check accuracy on test dataset
Got 9804/10000 with accuracy  98.04
----------------------------------------------------------------
"""

# imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


# network
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        # out_channels means the number of filters
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, stride=1,
                               padding=1, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, stride=1,
                               padding=1, kernel_size=3)
        # after conv2, maxPooling again,
        # according the formula output_size = (input_size+2*padding-kernel_size)//stride+1
        # so the fully-connected-nn size is 7, 16 is the channel
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x.shape[0] is the batch size
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameter
in_channels = 1
num_classes = 10
learning_rate = 3e-4
batch_size = 64
num_epochs = 10

# data load
train_dataset = datasets.MNIST(root='dataset/', train=True, download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='dataset/', train=False, download=True,
                              transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
model = CNN().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train model
for epoch in range(num_epochs):
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        # no need to flatten image data, nn.Conv2d function will handle the data
        scores = model(data)
        loss = criterion(scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(model, loader):
    if loader.dataset.train:
        print("check accuracy on train dataset")
    else:
        print("check accuracy on test dataset")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # scores batch_size x num_classes
            scores = model(x)
            # predicted is an index
            _, predicted = scores.max(1)
            num_correct += (predicted == y).sum()
            num_samples += predicted.size(0)
    model.train()
    print(f"Got {num_correct}/{num_samples} with accuracy {100.0 * num_correct / num_samples : .2f}")


check_accuracy(model, train_loader)
check_accuracy(model, test_loader)
