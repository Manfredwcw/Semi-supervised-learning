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

#stratified sampling
positive_percent = 0.0
negative_percent = 0.0
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
positive_percent = float(pn)/float(data_num)
negative_percent = float(nn)/float(data_num)

#parameters definition 
p = 500
n = 200
u = 4*p + 4*n
trainig_percent = 0.1
train_num = int(trainig_percent * data_num)
unlabeled_num = data_num - train_num

#shuffle examples in each set
p_data = p_data.sample(frac=1).reset_index(drop=True)
n_data = n_data.sample(frac=1).reset_index(drop=True)

#U: Unlabeled examples  L: Labeled trainig examples
#select 10% as trainig examples in positive and negative examples
L = p_data[:int(trainig_percent*pn)-1]
L = L.append(n_data[:int(trainig_percent*nn)-1])
L = L.sample(frac=1).reset_index(drop=True)
#put rest examples into U as unlabeled 
p_data = p_data[int(trainig_percent*pn):]
P_data = p_data.reset_index(drop=True)
n_data = n_data[int(trainig_percent*nn):]
n_data = n_data.reset_index(drop=True)
U_raw = p_data
U_raw = U_raw.append(n_data)
U_raw = U_raw.sample(frac=1).reset_index(drop=True)
U = U_raw

#create a pool U1 of examples by choosing u examples at random from U
U1 = U[0:u-1]
U1 = U1.reset_index(drop=True)
U = U[u:]
U = U.sample(frac=1).reset_index(drop=True)

#feature selection
'''
forest = ExtraTreesClassifier()
forest.fit(data_features_DF,data_class_DF)
print forest.feature_importances_
'''
'''
sort features according to its importance in classification.
(1.13.4.2. Tree-based feature selection
http://scikit-learn.org/stable/modules/feature_selection.html)
results(sorted):
  02：0.16642171
                 00：0.15250437  
  10：0.09517638
                 05：0.09210936
  01：0.04385695  
                 12：0.09138533
  04：0.08380449 
                 03：0.03662243  
  07：0.07951946
                 06：0.07489068  
  11：0.02871004
                 09：0.02255841
  13：0.0169574
                 08：0.01548301
select features every other one to ensure that 
every feature space sufficient to train a classifier.
so:
feature_space_1:01,02,04,07,10,11,13
feature_space_2:00,03,05,06,08,09,12
'''
feature_space_1 = ["Work-Class","fnlwgt","Education-Num","Relationship",
                   "Capital-gain","Capital-loss","Native-Country"]
feature_space_2 = ["Age","Education","Marital-Status","Occupation",
                   "Race","Sex","Hours-per-week"]
x1 = L[feature_space_1]
x2 = L[feature_space_2]
y = L[features_name[14]]
x1 = x1.astype(int)
x2 = x2.astype(int)
y = y.astype(int)
f1_score_h1 = []
f1_score_h2 = []
accuracy_h1 = []
accuracy_h2 = []

#k: iterations
k = 18
for i in range(k):
	bayes_h1 = GaussianNB()
	bayes_h1.fit(x1, y)
	bayes_h2 = GaussianNB()
	bayes_h2.fit(x2, y)

	prob1 = bayes_h1.predict_proba(U1[feature_space_1])
	sort_index_pos1 = []
	sort_index_neg1 = []
	#h1 label p positive and n negative examples
	rank_index_1 = np.argsort(-prob1, axis=0)
	for j in range(p):
		sort_index_pos1.append(rank_index_1[j][0])		
	temp = U1.loc[sort_index_pos1]
	temp['Earnings-Raw'] = temp['Earnings-Raw'].replace(1,0)
	#add h1 self-labeled positive examples to L
	L = L.append(temp)
	L = L.reset_index(drop=True)
	
	count = 0
	for j in range(n):
		if rank_index_1[count][1] in sort_index_pos1:
			j -= 1
			count += 1
		else:
			sort_index_neg1.append(rank_index_1[count][1])
			count += 1		
	temp = U1.loc[sort_index_neg1]
	temp['Earnings-Raw'] = temp['Earnings-Raw'].replace(0,1)
	#add h1 self-labeled negative examples to L
	L = L.append(temp)
	L = L.reset_index(drop=True)

	U1.drop(sort_index_pos1,inplace=True)
	U1.drop(sort_index_neg1,inplace=True)
	U1 = U1.reset_index(drop=True)

	prob2 = bayes_h2.predict_proba(U1[feature_space_2])
	#select p positive examples
	sort_index_pos2 = []
	sort_index_neg2 = []
	#h2 label p positive and n negative examples
	rank_index_2 = np.argsort(-prob2, axis=0)
	
	for j in range(p):
		sort_index_pos2.append(rank_index_2[j][0])	
	temp = U1.loc[sort_index_pos2]
	temp['Earnings-Raw'] = temp['Earnings-Raw'].replace(1,0)
	#add h2 self-labeled positive examples to L
	L = L.append(temp)
	L = L.reset_index(drop=True)

	count = 0
	for j in range(n):
		if rank_index_2[count][1] in sort_index_pos2:
			j -= 1
			count += 1
		else:
			sort_index_neg2.append(rank_index_2[count][1])
			count += 1
	temp = U1.loc[sort_index_neg2]
	temp['Earnings-Raw'] = temp['Earnings-Raw'].replace(0,1)
	#add h2 self-labeled negative examples to L
	L = L.append(temp)
	L = L.reset_index(drop=True)
	
	U1.drop(sort_index_pos2,inplace=True)
	U1.drop(sort_index_neg2,inplace=True)

	#add randomly 2p+2n examples from U to U1
	U1 = U1.append(U[:2*p+2*n-1])
	U1 = U1.reset_index(drop=True).astype(int)
	U = U.loc[2*p+2*n:]
	U = U.reset_index(drop=True).astype(int)

	#update x1,x2,y
	x1 = L[feature_space_1]
	x2 = L[feature_space_2]
	y = L[features_name[14]]
	x1 = x1.astype(int)
	x2 = x2.astype(int)
	y = y.astype(int)

	U_train = U.sample(frac=1).astype(int)
	y_true = U_train[features_name[14]]

	U_train_x1 = U_train[feature_space_1]
	y_pred_1 = bayes_h1.predict(U_train_x1)
	f1score_1 = f1_score(y_true, y_pred_1)
	accuracy_1 = accuracy_score(y_true, y_pred_1)

	U_train_x2 = U_train[feature_space_2]
	y_pred_2 = bayes_h2.predict(U_train_x2)
	f1score_2 = f1_score(y_true, y_pred_2)
	accuracy_2 = accuracy_score(y_true, y_pred_2)

	accuracy_h1.append(accuracy_1)
	accuracy_h2.append(accuracy_2)
	f1_score_h1.append(f1score_1)
	f1_score_h2.append(f1score_2)

	print "iteration:", i+1
	print "classifier_1: f1:", f1score_1, "acc:", accuracy_1
	print "classifier_2: f1:", f1score_2, "acc:", accuracy_2
	print "#####################################################"

#plot results
x = []
for i in range(k):
	x.append(i)	
plt.figure(1)

plt.subplot(211)
plt.axis([0,k,0,1])
plt.plot(x,accuracy_h1,'r',lw=2,label='h1')
plt.plot(x,accuracy_h2,'b',lw=2,label='h2')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(212)
plt.axis([0,k,0,1])
plt.plot(x,f1_score_h1,'r',x,f1_score_h2,'b',lw=2)
plt.xlabel('iterations')
plt.ylabel('f1_score')
plt.grid(True)
plt.tight_layout()
plt.show()