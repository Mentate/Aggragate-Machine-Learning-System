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
import math
from os import listdir
from os.path import isfile,join





data = pd.read_csv('Phones_gyroscope.csv')

target = data['gt']
lab_enc = preprocessing.LabelEncoder()
target = lab_enc.fit_transform(target)
testtarget = target[600000:3205431]#size of test set. Previously set to 5000. I tried to see how long 2.5 million records would take, it took too long. 
del data['Index']
del data['User']
del data['Model']
del data['Device']
del data['gt']
del data['Arrival_Time']
del data['Creation_Time']

print("Data imported")

result=""



def train_test(size):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #set up lists to hold prediction
        testset=data[600000:3205431]
        new_model=0
        dt_pre_B=[]
        rtree_pre_A=[]
        dt_pre_A =[]
        rtree_pre_B=[]
        dt_pre_all=[]
        rtree_pre_all=[]
        trainset=data[:size]

        #find our mid point
        mid = int(math.floor(size/2))

        #create subset A
        Adata=data[:mid]
        Atarget=target[:mid]

        #create subset B
        Bdata = data[mid:size]
        Btarget = target[mid:size]
        
        nn = KNeighborsClassifier(n_neighbors=3)
        dt = tree.DecisionTreeClassifier(max_depth=5)
        rtree = RandomForestClassifier(max_depth=5)

        print("Begin first round training")
        #first round training
        dt.fit(Adata,Atarget)
        rtree.fit(Bdata,Btarget)
        #now we have them predict the other half
        for i in range(0,mid):
            dt_pre_B.append(dt.predict(Bdata.iloc[i])[0])
            rtree_pre_A.append(rtree.predict(Adata.iloc[i])[0])

        #now we swap the subsets and classifiers
        dt = tree.DecisionTreeClassifier(max_depth=5)
        rtree = RandomForestClassifier(max_depth=5)

        print("Begin second round training")
        #second round of training
        dt.fit(Bdata,Btarget)
        rtree.fit(Adata,Atarget)

        #again, we predict the other half
        for i in range(0,mid):
            dt_pre_A.append(dt.predict(Adata.iloc[i])[0])
            rtree_pre_B.append(rtree.predict(Bdata.iloc[i])[0])
        dt_pre_all=dt_pre_A
        rtree_pre_all=rtree_pre_A
        for i in range(0,(mid)):
            dt_pre_all.append(dt_pre_B[i])
            rtree_pre_all.append(rtree_pre_B[i])
        rtree_pre_all = np.asarray(rtree_pre_all)
        dt_pre_all = np.asarray(dt_pre_all)
        trainset['rtree'] = pd.Series(rtree_pre_all, index=trainset.index)
        trainset['dt'] = pd.Series(dt_pre_all, index=trainset.index)
        print("Agraget Training set created")
        nn.fit(trainset,target[:size])
        print("Agraget Model Trained")
        dt_pre_test=[]
        rtree_pre_test = []

        
#begin providing test data to measure accuracy
        
                
        for i in range(0,2605431):#this needs to match the size of the test set
            current = i
            dt_pre_test.append(dt.predict(testset.iloc[current])[0])#just like in training we predict with dt and rtree
            rtree_pre_test.append(rtree.predict(testset.iloc[current])[0])
        dt_pre_test = np.asarray(dt_pre_test)
        rtree_pre_test = np.asarray(rtree_pre_test)
        testset['rtree'] = pd.Series(rtree_pre_test, index=testset.index)
        testset['dt'] = pd.Series(dt_pre_test, index=testset.index)#add the predictions as features

        print("Agraget test set generated")

        for i in range(0,2605431):#now we feed it into the final algorithm
                
            if testtarget[current] == nn.predict(testset.iloc[current]):
                                    new_model = new_model +1
        
    print("Training set Size: "+str(size))
    print("Correct: " + str(new_model))
    

    return(str(size) + "," + str(new_model))
result= ""+ str(train_test(500000)) #+ str(train_test(750000)) + str(train_test(1000000)) + str(train_test(1250000)) + str(train_test(1500000)) + str(train_test(1750000)) + str(train_test(2000000)) + str(train_test(2250000)) + str(train_test(2500000)) + str(train_test(2750000)) +str(train_test(3000000))

print(result)
