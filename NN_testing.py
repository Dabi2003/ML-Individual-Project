import torch
from torch import nn 
import numpy as np
from torch.utils.data import DataLoader
import torchmetrics
#loading data
X= torch.from_numpy(np.load("Data AI/X_data.npy")).type(torch.float)
label=torch.from_numpy(np.load("Data AI/label.npy")).type(torch.float)
groups=torch.from_numpy(np.load("Data AI/groups.npy")).type(torch.int)

print("Data loaded")
print(len(label),len(X))
# 80% of data (6100 samples) is used for training 20% is used for testing 

# For the training data I split data into 100 batches each containig  61 samples
X_train=DataLoader(dataset=X[:round(0.8*len(X))-1],batch_size=len(X[:round(0.8*len(X))])//100,shuffle=False)

y_train=label[:round(0.8*len(label)-1)]


X_test=X[6100:]
y_test=label[6100:]

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc
#Defining Neural network class
class NueralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack=nn.Sequential(
        nn.ReLU(),
        nn.Linear(in_features=1280, out_features=2000),
        nn.ReLU(),
        nn.Linear(in_features=2000, out_features=1),
        nn.ReLU()

        

   
        
        )
    def forward(self,x):
        return(self.stack(x))
    


#Declaring a Nueral network model
model=NueralNet()
model.load_state_dict(torch.load("NN1.pth"))
loss_func=torch.nn.MSELoss()
#testing
model.eval()
with torch.inference_mode():
    test_prediction=model(X_test.reshape(len(X_test),1280))
    test_prediction=test_prediction.reshape(len(test_prediction))
    test_loss=loss_func(test_prediction,y_test)
    acc_score=accuracy_fn(y_test,torch.round(test_prediction))
    print("done")