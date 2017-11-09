import _pickle as cPickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PredictionModel:

	LOGREG_MODEL = cPickle.load(open('models/logreg_model.p', 'rb'))
	LINEAR_SVC_MODEL = cPickle.load(open('models/linear_svc_model.p', 'rb'))
	KNN_MODEL = cPickle.load(open('models/knn_minmax_model.p', 'rb'))
	RF_MODEL = cPickle.load(open('models/random_forest_model.p', 'rb'))
	
	def predict_logreg(self, instance):
		return self.LOGREG_MODEL.predict(instance)

	def predict_linear_svc(self, instance):
		return self.LINEAR_SVC_MODEL.predict(instance)

	def predict_knn(self, instance):
		return self.KNN_MODEL.predict(instance)

	def predict_random_forest(self, instance):
		return self.RF_MODEL.predict(instance)

		
def test():
	pm = PredictionModel()

	df = pd.read_csv('./data/ny_hmda_2015_minmax.csv')

	x = np.array(df.drop(['action_taken'],1)) 
	y = np.array(df['action_taken'])
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

	print(x_test)
	print(pm.predict(x_test))