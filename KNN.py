import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
#Loading data 
X=(np.load("Data AI/X_data.npy"))[:,:1,:2]
X=X.reshape(len(X)*len(X[0]),2)
labels=np.load("Data AI/label.npy")
print(type(labels))


# This algorithm is so easy to implement and even easier to optimize due to the lack of parameters.
# 80% of data to train and 20% for testing
X_train,x_test,y_train,y_test=train_test_split(X,labels,train_size=0.8, random_state=0)
knn=KNeighborsClassifier(n_neighbors=11, p=2) #Using the Ecludian distance
knn.fit(X_train,y_train)
y_pred=knn.predict(x_test)
print(metrics.accuracy_score(y_pred,y_test))
print("whoo")