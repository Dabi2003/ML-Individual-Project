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


#Training model
optimizer=torch.optim.SGD(params=model.parameters(),lr=0.0001)
loss_func=torch.nn.MSELoss()

enumerate(y_train)
train_acc=0
losses=[]
accuracy=[]
for epoch in range(1000):
    model.train()
    train_loss=0
    train_acc=0
    for i,sample_i in enumerate(X_train):
            y_pred=model(sample_i.reshape(X_train.batch_size,1280))
            y_expected=y_train[i*61:(i*61)+61]
            y_pred=y_pred.reshape(len(y_pred))
            loss=loss_func(y_pred,y_expected)
            train_loss+=loss
            train_acc+=accuracy_fn(y_expected,torch.round(y_pred))
            #ALWAYS HAVE THIS AFTER CALULATING LOSS
            optimizer.zero_grad()
            clip_value=1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            loss.backward()
            optimizer.step()
    losses.append(train_loss/len(X_train)) 
    accuracy.append(train_acc/len(X_train))
    torch.save(model.state_dict(),f="NN1.pth")           
print("done")      

               


#Need a testing and validation group



