import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#Load the creditcard.csv using pandas
datainput = pd.read_csv('C:/Codes/debug/fds/FDS/creditcard.csv')
#separating features(X) and label(y)

X = datainput.iloc[:, :-1].values
Y = datainput.iloc[:, -1].values

#train_test_split method
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(max_depth=4)
classifier.fit(X_train,Y_train)
predicted=classifier.predict(X_test)
print("\npredicted values :\n",predicted)
#
# #Accuracy
DT = metrics.accuracy_score(Y_test, predicted) * 100
print("\nThe accuracy score using the DecisionTreeClassifier : ",DT)
#
# #Precision
print('precision')
# Precision = TP / (TP + FP) (Where TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative).
precision = precision_score(Y_test, predicted, pos_label=1)
print(precision_score(Y_test, predicted, pos_label=1))

#Recall
print('recall')
# Recall = TP / (TP + FN)
recall = recall_score(Y_test, predicted, pos_label=1)
print(recall_score(Y_test, predicted, pos_label=1))

#f1-score
print('f-Score')
# F - scores are a statistical method for determining accuracy accounting for both precision and recall.
fscore = f1_score(Y_test, predicted, pos_label=1)
print(f1_score(Y_test, predicted, pos_label=1))