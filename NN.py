import torch
from torch import nn 
import numpy as np

#loading data
X= torch.from_numpy(np.load("Data AI/X_data.npy")).type(torch.float)
label=torch.from_numpy(np.load("Data AI/label.npy")).type(torch.float)
groups=torch.from_numpy(np.load("Data AI/groups.npy")).type(torch.int)
print("Data loaded")

# 80% of data is used for training 20% is used for testing 
X_train=X[:round(0.8*len(X))]
y_train=label[:round(0.8*len(label))]

X_test=X[round(0.8*len(X)+1):]
y_test=label[round(0.8*len(label)+1):]


#Defining Neural network class
class NueralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1= nn.Linear(in_features=1280, out_features=10)
        self.layer_2=nn.Linear(in_features=10, out_features=10)
        self.layer_3=nn.Linear(in_features=10, out_features=1)
    
    def forward(self,x):
        return(self.layer_3(self.layer_2(self.layer_1(x))))
    


#Declaring a Nueral network model
model=NueralNet()

#Training model
optimizer=torch.optim.SGD(params=model.parameters(),lr=0.1)
loss_func=nn.CrossEntropyLoss()


for epoch in range(100):
    model.train()
    for sample_i in range(0,len(X_train)):
            optimizer.zero_grad()
            y_pred=model(X_train[sample_i].reshape([128*10]))
            y_expected=y_train[sample_i].reshape([1])
            loss=loss_func(y_pred,y_expected)
            loss.backward()
            optimizer.step()







