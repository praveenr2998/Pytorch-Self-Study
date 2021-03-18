import torch
import torch.nn as nn
from torch.optim import SGD



# SAVING A MODEL(state_dict)

torch.save(trained_model.to('cpu').state_dict(), '/home/praveen/Desktop/Computer Vision/model.pth')








# LOADING THE WEIGHTS OF SAVED MODEL

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



model = MyNeuralNet()

state_dict = torch.load("/home/praveen/Desktop/Computer Vision/model.pth")

model.load_state_dict(state_dict)

model.to('cuda')


val_x = [[10, 11]]

val_x = torch.tensor(val_x).float().to('cuda')

output = model(val_x)
print(output)
