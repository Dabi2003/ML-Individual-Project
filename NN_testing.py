import torch
from torch import nn ,tensor
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn  import metrics
from torchmetrics import AUROC
#loading data
X=torch.from_numpy(np.load("X_oversample_shuffled.npy")).type(torch.float)
label=torch.from_numpy(np.load("labels_oversample_shuffled.npy")).type(torch.int)
torch.manual_seed(50)
tags={}

for num in range(0,9):
    distribution=[0]*9
    distribution[num]=1
    tags[num]=distribution


# For the training data I split data into 3050 batches each containig  2 samples
X_train=DataLoader(dataset=X[:round(0.8*len(X))],batch_size=761,shuffle=False)

y_train=label[:round(0.8*len(label))]

X_test=DataLoader(dataset=X[9120:],batch_size=2283,shuffle=False)
y_test=label[9120:]

#print(X_test.batch_size)
distribution=torch.tensor([0]*9)
# Calculate accuracy (a classification metric)

AUC=[]

def accuracy_fn(y_pred,y_t):
    
    classes=[]
    labels=[]
    labels_true=[]
    for label in y_pred:
        classes.append(torch.argmax(label))
    classes=torch.from_numpy(np.array(classes))
    correct = metrics.accuracy_score(torch.detach(classes).numpy(),torch.detach(y_t).numpy()) # torch.eq() calculates where two tensors are equal
    acc = (correct) * 100 
    
    for l in classes:
        labels.append(tags[l.item()])
    
    labels=np.array(labels).astype(float)
    y_t_numpy=torch.detach(y_t.squeeze()).numpy()
    AUC.append(metrics.roc_auc_score(y_t_numpy,labels,multi_class="ovr")*100)
    
    return acc

#Defining Neural network class
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
#Declaring a Nueral network model
model=NueralNet()
#print(model.parameters())
model.load_state_dict(torch.load("NN1.pth"))
model.eval()
#testing


def Test(sample,test_loss,acc_score,i):
      test_prediction=model(sample).squeeze()
      y_to_compare=((y_test[i*X_test.batch_size:(i+1)*X_test.batch_size]).reshape(len(y_test[i*X_test.batch_size:(i+1)*X_test.batch_size]),1)).type(torch.LongTensor)
      acc_score+=accuracy_fn(test_prediction,y_to_compare)
    
      return(test_loss,acc_score)


test_loss=0
acc_score=0
acc=[]
losses=[]

with torch.inference_mode():
    for i,sample_i in enumerate(X_test):
        test_loss,acc_score=Test(sample_i.reshape(X_test.batch_size,128*10),test_loss,acc_score,i)
       

    print("Model: NN")
    average_accuracy=acc_score/(i+1)
    print("Test acuracy: ",average_accuracy)
    average_loss=test_loss/(i+1)
    print("AUC: ", np.sum(AUC)/len(AUC))
