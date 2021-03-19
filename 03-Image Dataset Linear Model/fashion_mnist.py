from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np 
from torchvision import datasets
from torch.optim import SGD
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"



# Downloading the dataset
data_path = "/home/praveen/Desktop/Computer Vision/FMNIST"
fmnist = datasets.FashionMNIST(data_path, download = True, train = True)

tr_images = fmnist.data
tr_targets = fmnist.targets




class CustomDataset(Dataset):
    def __init__(self, x, y):
        x = x.float()/255        # Scaling the data(image pixel values)
        x = x.view(-1, 28*28)
        self.x, self.y = x, y

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return x.to(device), y.to(device)
    
    def __len__(self):
        return len(self.x)




train = CustomDataset(tr_images, tr_targets)
trn_dl = DataLoader(train, batch_size=32, shuffle=True)



class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(28*28, 1000)
        self.activation = nn.ReLU()
        self.output = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output(x)
        return x


model = FashionMNISTModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr = 1e-2)



def train_batch(x, y, model, opt, loss_fn):
    model.train() 
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()



@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()



losses, accuracies = [], []
for epoch in range(5):
    print(epoch)
    epoch_losses, epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        epoch_losses.append(batch_loss)
    epoch_loss = np.array(epoch_losses).mean()
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        epoch_accuracies.extend(is_correct)
    epoch_accuracy = np.mean(epoch_accuracies)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)




epochs = np.arange(5)+1
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.legend()
plt.show()



