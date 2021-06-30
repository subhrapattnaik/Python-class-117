#Now, let's take a dataset and perform some logistic regression on it to see the accuracy of our model with the confusion matrix.


#Let's see how the age of the person increases the list of a heart attack, by using single variable logistic regression.
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  
import pandas as pd

df = pd.read_csv("./117/heart.csv")

print(df.head())

#--------------------------------------
#Let's see how the age of the person increases the chance of heart attach by single variable logistic regression
#from this data,we are going to use the age and target
#we are going to split this data into 75% and 25% to train our prediction model and then test it

from sklearn.model_selection import train_test_split 

age = df["age"]
heart_attack = df["target"]

#train_test_split() function splits the columns for training and testing purposes in ratio of 75% and 25% respectively
age_train, age_test, heart_attack_train, heart_attack_test = train_test_split(age, heart_attack, test_size = 0.25, random_state = 0)

#now train the prediction model on the data
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.reshape(age_train.ravel(), (len(age_train), 1))
Y = np.reshape(heart_attack_train.ravel(), (len(heart_attack_train), 1))


classifier = LogisticRegression(random_state = 0) 
classifier.fit(X, Y)

#----------------------------------------------
#now test the prediction model 

X_test = np.reshape(age_test.ravel(), (len(age_test), 1))
Y_test = np.reshape(heart_attack_test.ravel(), (len(heart_attack_test), 1))

heart_attack_prediction = classifier.predict(X_test)

predicted_values = []
for i in heart_attack_prediction:
  if i == 0:
    predicted_values.append("No")
  else:
    predicted_values.append("Yes")

actual_values = []
for i in Y_test.ravel():
  if i == 0:
    actual_values.append("No")
  else:
    actual_values.append("Yes")

    #---------------------------------------

    labels = ["Yes", "No"]

cm = confusion_matrix(actual_values, predicted_values, labels)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
plt.show()
#Observation:From here, we can see the following things -

#36 People both actually got a heart attack and were predicted to get a heart attack.
#16 People didn't get a heart attack and were also predicted to not get a heart attack.
#7 People actually got a heart attack while they were not predicted to get one.
#17 people were predicted to get a heart attack while they did not get one.



accuracy = 36 + 16 / 36 + 16 + 17 + 7
#accuracy = 52 / 76
#accuracy = 0.68421052631
print(accuracy)

#What if we use more variables, instead of just age? Will our model's accuracy increase? Let's see!