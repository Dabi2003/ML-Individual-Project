import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from operator import add, sub
import torch
from torch import nn
#Loading data 
X=np.load("X_oversample.npy")
X=X.reshape(len(X),len(X[0])*len(X[0][0]))
labels=np.load("labels_oversample.npy")
print(type(labels))

distribution=torch.tensor([0]*9)

X_train,x_test,y_train,y_test=train_test_split(X,labels,train_size=0.8, random_state=0)
# This algorithm is so easy to implement and even easier to optimize due to the lack of parameters.
# 80% of data to train and 20% for testing
loss=nn.CrossEntropyLoss()
def KNN(k):#Using the Ecludian distance
   
    knn=KNeighborsClassifier(n_neighbors=k, p=1) 
    knn.fit(X_train,y_train)
    y_pred=knn.predict(x_test)
    acc.append(metrics.accuracy_score(y_test,y_pred)*100)

k=2
sigma=1.46
count=1
succ=0
costs=[]
acc=[]
sigmas=[]


k_s=[]
for k in range(1,100):
    k_s.append(k)
    KNN(k)

acc=torch.tensor(acc)

print(k_s[torch.argmax(acc)])
plt.plot(k_s,acc)
plt.title("Accuray against K parameters")
plt.xlabel("k parameters")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



#for i in range(0,100):
    #k_m= round(k + sigma*np.random.randn())
    #if((succ/count)>(1/5)):
       #sigma_m= sigma*math.exp(1/3)
    #else:
       #sigma_m=sigma/math.exp(1/12)
     
   
        
   # if(k_m==0):
      # break
    
   # if(Cost(k_m)<Cost(k)):
       
       #succ+=1
      # k=k_m
       #sigma=sigma_m
    
    #count+=1
   # sigmas.append(sigma)
    #costs.append(Cost(k))
    
    
