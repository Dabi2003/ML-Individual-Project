import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
#Loading data 
X=(np.load("Data AI/X_data.npy"))[:,:1,:2]
X=X.reshape(len(X)*len(X[0]),2)
labels=np.load("Data AI/label.npy")
print(type(labels))
groups=np.load("Data AI/groups.npy")

#Training using SVM
X_train,x_test,y_train,y_test=train_test_split(X,labels,train_size=0.8, random_state=0)
k=svm.SVC(kernel="poly",C=1,decision_function_shape="ovo").fit(X_train,y_train)

#Creating mesh grid
x1_max,x1_min=X[:,0].max()+1,X[:,0].min()-1
x2_min,x2_max=X[:,1].min()-1,X[:,1].min()+1
x1,x2=np.meshgrid( np.arange(x1_min,x1_max,0.01),np.arange(x2_min,x2_max,0.01))

#Ploting samples in a 3D space
Z=k.predict(np.c_[x1.ravel(),x2.ravel()])

#plotting countors
Z=Z.reshape(x1.shape)
plt.contourf(x1,x2,Z,cmap="bone", alpha=0.7)
plt.scatter(X[:,0],X[:,1], cmap=plt.cm.PuBuGn, edgecolors='grey')
plt.xlabel("Feature1")
plt.ylabel("Feature2")

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.show()

print(k.score(x_test,y_test))





