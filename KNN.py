import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

#Loading data 
X=(np.load("X_oversample_shuffled.npy"))
X=X.reshape(len(X),len(X[0])*len(X[0][0]))
labels=np.load("labels_oversample_shuffled.npy")
print("Model: KNN")

distribution=torch.tensor([0]*9)

# This algorithm is so easy to implement and even easier to optimize due to the lack of parameters.
# 80% of data to train and 20% for testing
X_train,x_test,y_train,y_test=train_test_split(X,labels,train_size=0.8, random_state=0)
knn=KNeighborsClassifier(n_neighbors=1, p=2) #Using the Ecludian distance
knn.fit(X_train,y_train)
y_pred=knn.predict(x_test)
y_train_test=knn.predict(X_train)
print( "Test accuracuy: ",metrics.accuracy_score(y_pred,y_test)*100,"%")
#print( "Train accuracy: ",metrics.accuracy_score(y_train_test,y_train)*100)
l=knn.predict_proba(x_test)
print(" AUC: ",metrics.roc_auc_score(y_test,knn.predict_proba(x_test) ,multi_class="ovr")*100,"%")
l=knn.predict_proba(x_test)
