# I am using all the features( more data)

import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
#Loading data 
X=(np.load("X_oversample_shuffled.npy"))


#Making matrix 7626x1280 (7626 rows and 1280 coloumns)
X=X.reshape(len(X),len(X[0])*len(X[0][0]))

labels=np.load("labels_oversample_shuffled.npy")
print("Model SVM:")

#Training using SVM
X_train,x_test,y_train,y_test=train_test_split(X,labels,train_size=0.8, random_state=0)
k=svm.SVC(kernel="rbf",C=17, gamma=0.001,decision_function_shape="ovo" , probability=True).fit(X_train,y_train)
# Parameters wehere gotten from efficent robot paper
x1_min,x1_max=X[:,1].min()-1, X[:,1].max()+1
x2_min,x2_max=X[:,2].min()-1, X[:,2].max()+1
x1,x2=np.meshgrid(np.arange(x1_min,x1_max,0.001),np.arange(x2_min,x2_max,0.001))


#transforming samples into a higher dimension
Z=k.predict(x_test)
Z_train=k.predict(X_train)
#print(len(Z))
Z_origional=Z
Z=Z[:100]
#plotting countors
#reconstructing x1 and x2 to be the same size as Z
x1=x1[:10,:10]
x2=x2[:10,:10]
Z=Z.reshape(x1.shape)
plt.contourf(x1,x2,Z,cmap="plasma", alpha=0.7)
plt.scatter(X[:,0],X[:,1], c=labels,cmap=plt.cm.PuBuGn, edgecolors='red')
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
#plt.show()

#print("Training accuracy: ", metrics.accuracy_score(Z_train,y_train)*100)
print("Test accuracy :", metrics.accuracy_score(Z_origional,y_test)*100)
print(" AUC: ",metrics.roc_auc_score(y_test,k.predict_proba(x_test) ,multi_class="ovr")*100)
