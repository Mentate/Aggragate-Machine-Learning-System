#AG.py
import math
from multiprocessing import Process, Array
import pickle
from sklearn import tree
import warnings
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from skimage.io import imread
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import pandas as pd

class AGClassifier(object):
        Dec_T_1 = tree.DecisionTreeClassifier(max_depth=5)
	Dec_T_2 = tree.DecisionTreeClassifier(max_depth=5)
	Ran_F_1 = RandomForestClassifier(max_depth=5)
	Ran_F_2 = RandomForestClassifier(max_depth=5)
	Near_N = KNeighborsClassifier(n_neighbors=3)
	TrainingSet=[]
	SubsetA=[]
	SubsetB=[]
	TrainingTarget= []
	SubTargetA=[]
	SubTargetB=[]

	def __init__(self):
                garbage=""
		


	def classify(self,classifier,dataset,predict_result):
		for i in range(0,dataset.shape[0]):
                        with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
			        predict_result[i]=(classifier.predict(dataset.iloc[i])[0])

	def train(self,X,y):
		self.TrainingSet = X
		self.TrainingTarget = y
		size = X.shape[0]
		midpoint = int(math.floor((size/2)))
		self.SubsetA=X[:midpoint]
		self.SubsetB = X[midpoint:size]
		self.SubTargetA = self.TrainingTarget[0:midpoint]
		self.SubTargetB = self.TrainingTarget[midpoint:]
		self.Dec_T_1.fit(self.SubsetA,self.SubTargetA)
		self.Dec_T_2.fit(self.SubsetB,self.SubTargetB)
		self.Ran_F_1.fit(self.SubsetA,self.SubTargetA)
		self.Ran_F_2.fit(self.SubsetB,self.SubTargetB)

		DT_pre = []
		DT_pre_A = Array('i', range(midpoint))
		DT_pre_B = Array('i', range((size-midpoint)))
		RT_pre = []
		RT_pre_A = Array('i', range(midpoint))
		RT_pre_B = Array('i', range((size-midpoint)))

		jobs=[Process(target=self.classify, args=(self.Dec_T_1,self.SubsetB,DT_pre_B)), Process(target=self.classify, args=(self.Dec_T_2,self.SubsetA,DT_pre_A)), Process(target=self.classify, args=(self.Ran_F_1,self.SubsetB,RT_pre_B)), Process(target=self.classify, args=(self.Ran_F_2,self.SubsetA,RT_pre_A))]
		for j in jobs:
			j.start()
                        print("set trained")

		for j in jobs:
			j.join()
                for pred in DT_pre_A:
                        DT_pre.append(pred)
                for pred in DT_pre_B:
                        DT_pre.append(pred)
                for pred in RT_pre_A:
                        RT_pre.append(pred)
                for pred in RT_pre_B:
                        RT_pre.append(pred)
		DT_pre = np.asarray((DT_pre))
		RT_pre = np.asarray((RT_pre))

                
		self.TrainingSet['dt'] = pd.Series(DT_pre, index=self.TrainingSet.index)
		self.TrainingSet['rt'] = pd.Series(RT_pre, index=self.TrainingSet.index)
		self.Near_N.fit(self.TrainingSet,self.TrainingTarget)
		
