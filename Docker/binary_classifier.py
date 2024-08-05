
# Import PyTorch and matplotlib
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer=nn.Linear(in_features=1, 
                                      out_features=1)
    
    def forward(self,x):
        return self.linear_layer(x)
    

def create_dataset():
    # Create weight and bias
    weight = 0.7
    bias = 0.3

    # Create range values
    start = 0
    end = 1
    step = 0.02

    # Create X and y (features and labels)
    X = torch.arange(start, end, step).unsqueeze(dim=1) # without unsqueeze, errors will happen later on (shapes within linear layers)
    y = weight * X + bias 
    # Split data
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    return X_train,y_train,X_test,y_test

model_state_path='model.pth'
model=LinearRegressionModel()
model.load_state_dict(torch.load(model_state_path))

# Evaluate loaded model


X_train,y_train,X_test,y_test=create_dataset()

model.eval()
with torch.inference_mode():
    train_preds = model(X_train)
    test_preds=model(X_test)


loss_fn = nn.L1Loss()

train_loss=loss_fn(train_preds,y_train)
test_loss=loss_fn(test_preds,y_test)

print(f'Train loss : {train_loss}')
print(f'Test loss : {test_loss}')