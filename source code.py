"Loading the Email spam classification dataset CSV from Keggle"
"https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv"


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Reads the dataset emails.csv
dataset = pd.read_csv('emails.csv')

# Removes the column EmailNo since it is not necessary for this application
# it only shows the indexes of the emails in the dataset
dataset = dataset.drop(columns=['Email No.'], axis=1)
# Removes the prediction column from X since it represents the input data values
X = dataset.drop('Prediction', axis=1)

# Sets y to be the output since prediction column represents if the email is
# a spam or not spam
y = dataset['Prediction']

# Training and testing split of 80/20 with a random_state of 50
# random state is used so that we get consistent results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 50)

# Creates the Gaussian Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

# Calculates the accuracy,precesion,recall and f1 scores using predictions and actual results
NB_accuracy = accuracy_score(y_test, y_pred)
NB_precision = precision_score(y_test, y_pred)
NB_recall = recall_score(y_test, y_pred)
NB_F1 = f1_score(y_test, y_pred)

# Prints out the results both in text and in bar graph
print("Naive Bayes Accuracy:" , NB_accuracy)
print("Naive Bayes Precision:" , NB_precision)
print("Naive Bayes Recall: ", NB_recall)
print("Naive Bayes F1: ", NB_F1)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
scores = [NB_accuracy, NB_precision, NB_recall, NB_F1]
plt.bar(metrics, scores, color='blue')
plt.ylim(0.8, 1)
plt.title('Naive Bayes Performance Metrics')
plt.ylabel('Scores')
plt.xlabel('Metrics')
plt.show()

# Creates a logistic regresion model with increased max iteration
LG_model = LogisticRegression(max_iter = 700)
LG_model.fit(X_train, y_train)
y_pred_log_reg = LG_model.predict(X_test)

# Calculates the accuracy,precision,recall and f1 scores
LG_accuracy = accuracy_score(y_test, y_pred_log_reg)
LG_precision = precision_score(y_test, y_pred_log_reg)
LG_recall= recall_score(y_test, y_pred_log_reg)
LG_F1 = f1_score(y_test, y_pred_log_reg)

# Prints out the results in text and as bar graphs
print("Logistic Regression Accuracy:" , LG_accuracy)
print("Logistic Regression Precision:" , LG_precision)
print("Logistic Regression Recall:" , LG_recall)
print("Logistic Regression F1:" , LG_F1)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
scores = [LG_accuracy, LG_precision, LG_recall, LG_F1]
plt.bar(metrics, scores, color='blue')
plt.ylim(0.8, 1)
plt.title('Logistic Regression Performance Metrics')
plt.ylabel('Scores')
plt.xlabel('Metrics')
plt.show()