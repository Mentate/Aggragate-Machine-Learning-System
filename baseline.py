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

from os import listdir
from os.path import isfile,join





data = pd.read_csv('Phones_accelerometer.csv')

target = data['gt']
lab_enc = preprocessing.LabelEncoder()
target = lab_enc.fit_transform(target)
del data['Index']
del data['User']
del data['Model']
del data['Device']
del data['gt']
del data['Arrival_Time']
del data['Creation_Time']
print("data imported")

result=""
print(data.shape)


def train_test(size):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = data[:size]
        y=target[:size]
        nn_correct=0
        DT_correct=0
        Rtree_correct=0
        nn = KNeighborsClassifier(n_neighbors=3)
        dt = tree.DecisionTreeClassifier(max_depth=5)
        rtree = RandomForestClassifier(max_depth=5)
        nn.fit(X,y)
        #print("nn trained")
        dt.fit(X,y)
        #print("Decision Tree trained")
        rtree.fit(X,y)
        #print("Random tree trained")
        for i in range(3200431,3205431):
            current = i
            if target[current] == dt.predict(data.iloc[current]):
                                    DT_correct= DT_correct + 1
            if target[current] == nn.predict(data.iloc[current]):
                                    nn_correct = nn_correct +1
            if target[current] == rtree.predict(data.iloc[current]):
                                    Rtree_correct = Rtree_correct +1
    print("Set Complete")

    return(str(size) + "," + str(nn_correct) + "," + str(DT_correct) + "," + str(Rtree_correct) + "\n")
result= ""+str(train_test(500000)) + str(train_test(750000)) + str(train_test(1000000)) + str(train_test(1250000)) + str(train_test(1500000)) + str(train_test(1750000)) + str(train_test(2000000)) + str(train_test(2250000)) + str(train_test(2500000)) + str(train_test(2750000)) +str(train_test(3000000))

print(result)
