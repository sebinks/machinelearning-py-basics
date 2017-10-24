from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris=load_iris()

features=iris.data
labels=iris.target

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=.5)

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)
# fitting the model
knn.fit(x_train, y_train)


# predict the response
pred = knn.predict(x_test)


from sklearn.metrics import accuracy_score
print("Accuracy=",accuracy_score(y_test,pred))

