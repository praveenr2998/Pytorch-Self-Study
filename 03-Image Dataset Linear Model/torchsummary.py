# !pip install torch_summary - PIP command

from torchsummary import summary
model, loss_fn, optimizer = get_model()
summary(model, X_train)