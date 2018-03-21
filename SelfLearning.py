# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

file_name_path = "adult.csv"
features_name = ['Age','Work-Class','fnlwgt','Education',
'Education-Num','Marital-Status','Occupation','Relationship',
'Race','Sex','Capital-gain','Capital-loss','Hours-per-week',
'Native-Country','Earnings-Raw']

#data collection
raw_data = pd.read_csv(file_name_path,header=None,
	names=["Age","Work-Class","fnlwgt","Education",
	"Education-Num","Marital-Status","Occupation",
	"Relationship","Race","Sex","Capital-gain",
	"Capital-loss","Hours-per-week","Native-Country",
	"Earnings-Raw"])	
raw_data.dropna(how='all',inplace=True)
data_num = len(raw_data)

#data preprocessing(convert string to integer)
#class Work-Class
work_encoder = LabelEncoder()
work_col = raw_data["Work-Class"]
work_col_encoded = work_encoder.fit_transform(work_col)
raw_data["Work-Class"] = work_col_encoded
#class Education
edu_encoder =LabelEncoder()
edu_col = raw_data["Education"]
edu_col_encoded = edu_encoder.fit_transform(edu_col)
raw_data["Education"] = edu_col_encoded
#class Marital-Status
mar_encoder =LabelEncoder()
mar_col = raw_data["Marital-Status"]
mar_col_encoded = mar_encoder.fit_transform(mar_col)
raw_data["Marital-Status"] = mar_col_encoded
#class Occupation
occ_encoder =LabelEncoder()
occ_col = raw_data["Occupation"]
occ_col_encoded = occ_encoder.fit_transform(occ_col)
raw_data["Occupation"] = occ_col_encoded
#class Relationship
rel_encoder =LabelEncoder()
rel_col = raw_data["Relationship"]
rel_col_encoded = rel_encoder.fit_transform(rel_col)
raw_data["Relationship"] = rel_col_encoded
#class Race
rac_encoder =LabelEncoder()
rac_col = raw_data["Race"]
rac_col_encoded = rac_encoder.fit_transform(rac_col)
raw_data["Race"] = rac_col_encoded
#class Sex
sex_encoder =LabelEncoder()
sex_col = raw_data["Sex"]
sex_col_encoded = sex_encoder.fit_transform(sex_col)
raw_data["Sex"] = sex_col_encoded
#class Native-Country
nat_encoder =LabelEncoder()
nat_col = raw_data["Native-Country"]
nat_col_encoded = nat_encoder.fit_transform(nat_col)
raw_data["Native-Country"] = nat_col_encoded
#class Earnings-Raw
#"<=50":0(positive);">50":1(negative)
ear_encoder =LabelEncoder()
ear_col = raw_data["Earnings-Raw"]
ear_col_encoded = ear_encoder.fit_transform(ear_col)
raw_data["Earnings-Raw"] = ear_col_encoded
#print ear_encoder.classes_

#initiate data features and classes 
data_features_DF = raw_data[features_name[0:14]]
data_class_DF = raw_data['Earnings-Raw']

#n_data: negative datas  p_data: positive datas
n_data = pd.DataFrame(columns=["Age","Work-Class",
	"fnlwgt","Education","Education-Num","Marital-Status",
	"Occupation","Relationship","Race","Sex","Capital-gain",
	"Capital-loss","Hours-per-week","Native-Country",
	"Earnings-Raw"]) 
p_data = pd.DataFrame(columns=["Age","Work-Class",
	"fnlwgt","Education","Education-Num","Marital-Status",
	"Occupation","Relationship","Race","Sex","Capital-gain",
	"Capital-loss","Hours-per-week","Native-Country",
	"Earnings-Raw"]) 


trainig_percent = 0.1
p = 500
n = 200
#stratified sampling
#pn:positive numbers nn:negative numbers
pn = nn = 0
for i in range(data_num):
	if data_class_DF[i] == 0:
		pn+=1
		p_data = p_data.append(raw_data.loc[i])
	elif data_class_DF[i] == 1:
		nn+=1	
		n_data = n_data.append(raw_data.loc[i])
	else:
		pass

#U: Unlabeled examples  L: Labeled trainig examples
#select 10% as trainig examples in positive and negative examples
L = p_data[:int(trainig_percent*pn)-1]
L = L.append(n_data[:int(trainig_percent*nn)-1])
L = L.sample(frac=1).reset_index(drop=True)
L = L.astype(int)

#put rest examples into U as unlabeled 
p_data = p_data[int(trainig_percent*pn):]
n_data = n_data[int(trainig_percent*nn):]
U = p_data
U= U.append(n_data)
U = U.sample(frac=1).reset_index(drop=True)
U = U.astype(int)
U_test = U

accuracy = []
f1score = []
bayes_classifier = GaussianNB()
it = 0
for i in range(40):
	bayes_classifier.fit(L[features_name[0:14]], L[features_name[14]])
	prob1 = bayes_classifier.predict_proba(U[features_name[0:14]])
	sort_index_pos1 = []
	sort_index_neg1 = []
	#classifier label p positive and n negative examples
	rank_index_1 = np.argsort(-prob1, axis=0)
	for j in range(p):
		sort_index_pos1.append(rank_index_1[j][0])		
	temp = U.loc[sort_index_pos1]
	temp['Earnings-Raw'] = temp['Earnings-Raw'].replace(1,0)
	#add h1 self-labeled positive examples to L
	L = L.append(temp)
	L = L.reset_index(drop=True).astype(int)
	
	count = 0
	for j in range(n):
		if rank_index_1[count][1] in sort_index_pos1:
			j -= 1
			count += 1
		else:
			sort_index_neg1.append(rank_index_1[count][1])
			count += 1		
	temp = U.loc[sort_index_neg1]
	temp['Earnings-Raw'] = temp['Earnings-Raw'].replace(0,1)
	#add h1 self-labeled negative examples to L
	L = L.append(temp)
	L = L.reset_index(drop=True)
	L = L.astype(int)

	U.drop(sort_index_pos1,inplace=True)
	U.drop(sort_index_neg1,inplace=True)
	U = U.reset_index(drop=True)
	U = U.astype(int)  
	
	acc = accuracy_score(U_test[features_name[14]], bayes_classifier.predict(U_test[features_name[0:14]]))
	f1 = f1_score(U_test[features_name[14]], bayes_classifier.predict(U_test[features_name[0:14]]))
	accuracy.append(acc)
	f1score.append(f1)
	it += 1
	print "iterations:",it
	print "accuracy:",acc
	print "f1score:",f1
	print "#####################################"

#plot results
x = []
for i in range(it):
	x.append(i)	
plt.figure(1)

plt.subplot(211)
plt.axis([0,it,min(accuracy)-0.01,max(accuracy)+0.01])
plt.plot(x,accuracy,'r',lw=2,label='h1')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.grid(True)

plt.subplot(212)
plt.axis([0,it,min(f1score)-0.01,max(f1score)+0.01])
plt.plot(x,f1score,'b',lw=2)
plt.xlabel('iterations')
plt.ylabel('f1_score')
plt.grid(True)
plt.tight_layout()
plt.show()
