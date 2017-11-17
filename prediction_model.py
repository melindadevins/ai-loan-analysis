__author__ = "Vincent"

import _pickle as cPickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statistics import mode
from keras.models import model_from_json

class PredictionModel:

	LOGREG_MODEL = cPickle.load(open('models/logreg_model.p', 'rb'))
	LINEAR_SVC_MODEL = cPickle.load(open('models/linear_svc_model.p', 'rb'))
	KNN_MODEL = cPickle.load(open('models/knn_minmax_model.p', 'rb'))
	RF_MODEL = cPickle.load(open('models/random_forest_model.p', 'rb'))
	SVC_MODEL = cPickle.load(open('models/svc_model.p', 'rb'))
	NB_MODEL = cPickle.load(open('models/gaussian_nb_model.p','rb'))

	def __init__(self):
		nn_file = open('models/model.json', 'r')
		LOADED_MODEL_NN = nn_file.read()
		nn_file.close()
		self.NN_MODEL = model_from_json(LOADED_MODEL_NN)
		self.NN_MODEL.load_weights('models/model.h5')
		self.NN_MODEL.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	def predict_logreg(self, instance):
		return self.LOGREG_MODEL.predict(instance)

	def predict_linear_svc(self, instance):
		return self.LINEAR_SVC_MODEL.predict(instance)

	def predict_knn(self, instance):
		return self.KNN_MODEL.predict(instance)

	def predict_random_forest(self, instance):
		return self.RF_MODEL.predict(instance)

	def predict_svc(self, instance):
		return self.SVC_MODEL.predict(instance)

	def predict_naive_bayes(self, instance):
		return self.NB_MODEL.predict(instance)

	def predict_nerual_network(self, instance):
		return self.NN_MODEL.predict(instance)[0]

	def get_nn_prediction(self, id=1):
		df = pd.read_csv('./data/demo_predict_normalize.csv')
		x = np.array(df.drop(['action_taken'],1)) 
		return self.predict_nerual_network(x)[0]






def test_combine_models():

	pm = PredictionModel()

	df = pd.read_csv('./data/ny_hmda_2015_minmax.csv')

	x = np.array(df.drop(['action_taken'],1)) 

	y = np.array(df['action_taken'])

	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

	# print(x_test)
	l1 = list(pm.predict_logreg(x_test))
	print('finish one')
	l2 = list(pm.predict_linear_svc(x_test))
	print('finish one')
	l3 = list(pm.predict_knn(x_test))
	print('finish one')
	l4 = list(pm.predict_random_forest(x_test))
	print('finish one')
	l5 = list(pm.predict_svc(x_test))
	print('finish one')
	n = 0
	for i in range(len(l1)):
		guess = mode([l1[i],l2[i],l3[i],l4[i],l5[i]])
		if guess == y_test[i]:
			n += 1
	print(float(n)/len(y_test))
#0.84995484967598

# test_combine_models()
def test_nn():
	pm = PredictionModel()
	df = pd.read_csv('./data/demo_predict_normalize.csv')
	x = np.array(df.drop(['action_taken'],1)) 
	y = np.array(df['action_taken'])
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
	print([x[0] for x in pm.predict_nerual_network(x)])
	print(list(y))

pm = PredictionModel()
print(pm.get_nn_prediction())
