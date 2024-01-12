# imports
import torch
import torch.nn as nn  # loss func
import torch.nn.functional as F  # activation
import torch.optim as optim  # optimizer
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# neural network
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameter
input_size = 784
hidden_size = 50
num_classes = 10
batch_size = 128
learning_rate = 0.001
num_epochs = 10

# load data
train_dataset = datasets.MNIST(root="dataset/", train=True, download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="dataset/", train=False, download=True,
                              transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
model = NN(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train model
for epoch in range(num_epochs):
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)
        scores = model(data)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# check accuracy

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)
            scores = model(data)
            _, predicted = scores.max(1)
            num_correct += (target == predicted).sum()
            num_samples += predicted.size(0)
    model.train()
    print(f"Got {num_correct}/{num_samples} with accuracy {100.0 * num_correct / num_samples:.2f}")


check_accuracy(model, train_loader)
check_accuracy(model, test_loader)
