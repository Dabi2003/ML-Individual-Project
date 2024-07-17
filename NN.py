import torch
from torch import nn 
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torchmetrics
import matplotlib.pyplot as plt
import random as rand
import imblearn
from sklearn import metrics

#loading data
#X=torch.from_numpy(np.load("Data AI/X_data.npy")).type(torch.float)
X=torch.from_numpy(np.load("X_oversample_shuffled.npy")).type(torch.float)
labels=torch.from_numpy(np.load("labels_oversample_shuffled.npy")).type(torch.int)

torch.manual_seed(80)


print(labels.shape)
#groups=torch.from_numpy(np.load("Data AI/groups.npy")).type(torch.int)
tags={} 


for num in range(0,9):
    distribution=[0]*9
    distribution[num]=1
    tags[num]=distribution
#print(len(groups))
print("Data loaded")

# Calculate accuracy (a classification metric)
def accuracy_fn(y_pred,y_t):
    classes=[]
    for item in y_pred:
        classes.append(torch.argmax(item))
    classes=torch.from_numpy(np.array(classes))
    correct = metrics.accuracy_score(torch.detach(classes).numpy(),torch.detach(y_t).numpy())
    acc = (correct) * 100 
    return acc
# Neural network Archituectuer Input year 1280 neurons, output layer 1 neuron, ->1280->1280->1280->10->1
class NueralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack=nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features=1280, out_features=500),
        nn.ReLU(),
        nn.BatchNorm1d(500),
        nn.Dropout(0.2),
        nn.Linear(in_features=500,out_features=500),
        nn.ReLU(),
        nn.BatchNorm1d(500),
        nn.Dropout(0.2),
        nn.Linear(in_features=500,out_features=500),
        nn.ReLU(),
        nn.BatchNorm1d(500),
        nn.Dropout(0.2),
        nn.Linear(in_features=500,out_features=500),
        nn.ReLU(),
        nn.BatchNorm1d(500),
        nn.Dropout(0.3),
        nn.Linear(in_features=500,out_features=9),
        )
    def forward(self,x):
        return(self.stack(x))


#Training Sequence
def Train(sample,train_loss,train_acc,i):
          
            y_pred=model(sample).squeeze()

            y_to_compare=(((y_train[i*(X_train.batch_size):(i+1)*X_train.batch_size]).reshape(len(y_train[i*X_train.batch_size:(i+1)*X_train.batch_size]),1)).type(torch.LongTensor)).flatten()
            
            one_hots=[]
            for l in y_to_compare:
                one_hots.append(tags[l.item()])
            #computing loss
            one_hots=torch.from_numpy(np.array(one_hots)).type(torch.float)
            loss=loss_func(y_pred,one_hots)
            train_loss+=loss
        
            train_acc+=accuracy_fn(y_pred,y_to_compare)
           
            #ALWAYS HAVE THIS AFTER CALULATING LOSS
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return(train_loss,train_acc)


def Train_last(sample,train_loss,train_acc,i):
          
            y_pred=model(sample).squeeze()

            y_to_compare=((y_train[i*(X_train.batch_size):(i+1)*X_train.batch_size]).reshape(len(y_train[i*X_train.batch_size:(i+1)*X_train.batch_size]),1)).type(torch.LongTensor)
            #computing loss
            loss=loss_func(y_pred,y_to_compare.flatten())
            train_loss+=loss
        
            train_acc+=accuracy_fn(y_pred,y_to_compare)
        
            return(train_loss,train_acc)


#
l=0
r=3991
acc=[]
losses=[]

X_train=DataLoader(dataset=X[:9120],batch_size=30,shuffle=False)
y_train=labels[:9120]

#Sanity checking
print(X_train.batch_size)
print(len(y_train))
    
#Creating one hot encoding tags
for num in range(0,9):
    distribution=[0]*9
    distribution[num]=1
    tags[num]=distribution



    #Declaring a Nueral network model
model=NueralNet()
 


#Defining loss function and learning rate
optimizer=torch.optim.SGD(params=model.parameters(),lr=0.05)
loss_func=torch.nn.CrossEntropyLoss()

#Training
losses=[]
accuracy=[]
epochs=[]
distributions=[]
model.train()
for epoch in range(101):
    train_loss=0
    train_acc=0
    for i,sample_i in enumerate(X_train):
        train_loss,train_acc=Train(sample_i.reshape(X_train.batch_size,128*10),train_loss,train_acc,i)

    #print(i)
    average_loss=(train_loss/(i+1))
    average_accuracy=(train_acc/(i+1))
    print("Epoch ",epoch," Model training accuracy: ", average_accuracy)
    losses.append(average_loss.detach().numpy())
    accuracy.append(average_accuracy)
    epochs.append(epoch)

plt.plot(np.array(epochs),np.array(losses))
plt.xlabel("epoch")
plt.ylabel("Training loss") 
plt.show()     

plt.plot(np.array(epochs),np.array(accuracy))
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.show()       
print("done")

model.eval()
torch.save(model.state_dict(),f="NN1.pth") 
print(distributions)


