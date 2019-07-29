from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix


import pandas as pd
from pandas import Series
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import matplotlib.pyplot as plt

df = pd.read_csv('bug_report.csv',encoding='utf-8',error_bad_lines=False)


f=open('bug_report.csv')
stop_word_removal=f.read()
f.close()
stop_words=set(stopwords.words("english"))
filtered=[""]
print("stop words removed..")

csv_file = 'bug_report.csv'
txt_file = 'bug_report.txt'
with open(txt_file, 'w') as my_output_file:
    with open(csv_file, 'r') as my_input_file:
        for row in csv.reader(my_input_file):
            my_output_file.write(" ".join(row)+'\n')
			
			
			  

words = df['Summary'].str.split()


X = df[['Summary']]
Y = df[['component']]
x_train, x_test, y_train, y_test = train_test_split(X['Summary'],Y['component'],test_size=0.3,random_state=1)


NB = MultinomialNB(alpha=.6)
NB.fit(x_train,y_train)
NB_predicted = NB.predict(x_test)
NBC = NB.score(x_test,y_test)
print("\nAccuracy of Naive Bayes Classifier: ",NBC*100,"%\n")


S = LinearSVC()
S.fit(x_train,y_train)
S_predicted = S.predict(x_test)
SC = S.score(x_test,y_test)
print("\nAccuracy of Linear Support Vector Classifier: ",SC*100,"%\n")
