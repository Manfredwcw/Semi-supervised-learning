# Semi-supervised-learning
Co-Training:

1.feature selection:
sort features according to its importance in classification.(1.13.4.2. Tree-based feature selection http://scikit-learn.org/stable/modules/feature_selection.html)

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
                 
select features every other one to ensure that every feature space sufficient to train a classifier.

so:

feature_space_1:01,02,04,07,10,11,13

("Work-Class","fnlwgt","Education-Num","Relationship","Capital-gain","Capital-loss","Native-Country")

feature_space_2:00,03,05,06,08,09,12

("Age","Education","Marital-Status","Occupation","Race","Sex","Hours-per-week")

2.classifiers:

in this task two bayes classifiers are used to train datas.

3.runtime:

it takes about 3 min after to get the result after running the program.

plot the accuracy and f1scores of every iteration for each classifier.
