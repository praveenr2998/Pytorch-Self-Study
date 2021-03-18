import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Input and output
x = [[1,2], [3,4], [5,6], [7,8]]
y = [[3], [7], [11], [15]]

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Moving tensors(input and outputs) to GPU
X = X.to(device)
Y = Y.to(device)



# Defining custom dataset class 
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]        # ix here is the index that is fetched


data = CustomDataset(X, Y)

dataloader = DataLoader(data, batch_size=2, shuffle=True)






# A basic neural network architecture
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8,1)


    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)

        return x




mynet = MyNeuralNet().to(device)

loss_func = nn.MSELoss()          # Mean squared error

opt = SGD(mynet.parameters(), lr = 0.001)

loss_history = []


# Training
for i in range(50):
    for dat in dataloader:
        x, y = dat
        opt.zero_grad()                             # deletes the previously calculated gradients
        loss_value = loss_func(mynet(x), y)         # calculate loss
        loss_value.backward()                       # perform backpropagation ()
        opt.step()                                  # Update weights according according to gradients computed
    loss_history.append(loss_value)
    print("End of ",i+1," epoch")



# Plotting the loss
#plt.plot(loss_history)
#plt.xlabel('epochs')
#plt.ylabel('loss value')
#plt.show()





# Creating a validation dataset and testing the model

val_x = [[10, 11]]

val_x = torch.tensor(val_x).float().to(device)

output = mynet(val_x)

print("The output generated for the validation data is", output)



