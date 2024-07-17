# I am using all the features( more data)

import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import math
#Loading data 
X=(np.load("X_oversample.npy"))


#Making matrix 7626x1280 (7626 rows and 1280 coloumns)
X=X.reshape(len(X),len(X[0])*len(X[0][0]))

distribution=torch.tensor([0]*9)
loss=nn.CrossEntropyLoss()
labels=np.load("labels_oversample.npy")
print(type(labels))
X_train,x_test,y_train,y_test=train_test_split(X,labels,train_size=0.8, random_state=0)
def Cost(c):#Using the Ecludian distance
    error=[]
    s=svm.SVC(kernel="rbf",C=c, gamma=0.001,decision_function_shape="ovo").fit(X_train,y_train)
    y_pred=s.predict(x_test)
    for i in range(0,len(y_test)):
        
        #Defininging predictaed label
        distribution[round(y_pred[i])]=1
        y1=distribution.type(torch.float)
        distribution[round(y_pred[i])]=0

        #Defining test label 
        distribution[int(y_test[i])]=1
        y2=distribution.type(torch.float)
        distribution[int(y_test[i])]=0
    
        error.append(abs(y_test[i]-y_pred[i]))

    acc.append(metrics.accuracy_score(y_pred,y_test)*100)
    return(np.mean(error))
    


costs=[]

acc=[]
gamas=[]
# Optimizing regularization parameter
c=1
while c<21:
    gamas.append(c)
    costs.append(Cost(c)) 
    c=c+1

#Obtaining metrics
acc=torch.tensor(acc)
costs=torch.tensor(costs)
print(torch.argmin(costs))
print(torch.argmax(acc))


c=torch.argmin(costs).item()
plt.title("losses against regulization paramter")
plt.xlabel("parameter value")
plt.ylabel("losses")
plt.plot(gamas,costs)
plt.legend()
plt.show()

plt.title("Accuracy(%) against regulization parameters")
plt.xlabel("parameter value")
plt.ylabel("Accuracy")
plt.plot(gamas,acc)
plt.legend()
plt.show()
print(c)
 
 
#Training using SVM
k=svm.SVC(kernel="rbf",C=17, gamma=0.001,decision_function_shape="ovo").fit(X_train,y_train)
# Parameters wehere gotten from efficent robot paper
x1_min,x1_max=X[:,1].min()-1, X[:,1].max()+1
x2_min,x2_max=X[:,2].min()-1, X[:,2].max()+1
x1,x2=np.meshgrid(np.arange(x1_min,x1_max,0.001),np.arange(x2_min,x2_max,0.001))


#transforming samples into a higher dimension
Z=k.predict(x_test)
Z_origional=Z
Z=Z[:100]
#plotting countors
#reconstructing x1 and x2 to be the same size as Z
x1=x1[:10,:10]
x2=x2[:10,:10]
Z=Z.reshape(x1.shape)
plt.contourf(x1,x2,Z,cmap="plasma", alpha=0.7)
plt.scatter(X[:,1],X[:,2], c=labels,cmap="gist_gray", edgecolors='red')
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.show()


print(metrics.accuracy_score(Z_origional,y_test))