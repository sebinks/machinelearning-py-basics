from scipy.spatial import distance
def eucli(a,b):
	return distance.euclidean(a,b)

class myKNN():
#Fit method of classsifier
	def fit(self,x_train,y_train):
		self.x_train=x_train
		self.y_train=y_train
#Predict method of classifier
	def predict(self,x_test):
		predictions=[]
		for row in x_test:
			labels=self.closest(row)
			predictions.append(labels)
		return predictions
#closest distance
	def closest(self,row):
		best_dist=eucli(row,self.x_train[0])
		best_index=0
		for i in range(1,len(self.x_train)):
			dist=eucli(row,self.x_train[i])
			if dist<best_dist:
				best_dist=dist
				best_index=1
		return self.y_train[best_index]


from sklearn.datasets import load_iris

from  sklearn.linear_model import LogisticRegression

iris=load_iris()
x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)


clf=LogisticRegression()
clf.fit(x_train,y_train)

p=clf.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy=",accuracy_score(y_test,p))
