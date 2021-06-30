#plot the data on a heatmap and visually analyze the accuracy of the prediction model of multilienear regression

#What if we use more variables, instead of just age? Will our model's accuracy increase? Let's see!


#In our data, cp stands for chest pain and chol stands for cholestrol. thalach stands for Maximum heart rate achieved. Let's include these along with the gender of the person.
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import numpy as np

df = pd.read_csv("./117/heart.csv")

print(df.head())




factors = df[["age", "sex", "cp", "chol", "thalach"]]
heart_attack = df["target"]

factors_train, factors_test, heart_attack_train, heart_attack_test = train_test_split(factors, heart_attack, test_size = 0.25, random_state = 0)

#Since all of age, sex, cp and chol have different measurement units, let's make them scaler to analyse them well.

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

factors_train = sc_x.fit_transform(factors_train)  
factors_test = sc_x.transform(factors_test)

#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   #intercept_scaling=1, l1_ratio=None, max_iter=100,
                   #multi_class='auto', n_jobs=None, penalty='l2',
                   #random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   #warm_start=False)
classifier2 = LogisticRegression(random_state = 0) 
classifier2.fit(factors_train, heart_attack_train)



heart_attack_prediction_1 = classifier2.predict(factors_test)

predicted_values_1 = []
for i in heart_attack_prediction_1:
  if i == 0:
    predicted_values_1.append("No")
  else:
    predicted_values_1.append("Yes")

actual_values_1 = []
for i in heart_attack_test.ravel():
  if i == 0:
    actual_values_1.append("No")
  else:
    actual_values_1.append("Yes")

labels = ["Yes", "No"]
cm = confusion_matrix(actual_values_1, predicted_values_1, labels)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
plt.show()
#From here, we can see the following things -

#33 People both actually got a heart attack and were predicted to get a heart attack.
#23 People didn't get a heart attack and were also predicted to not get a heart attack.
#10 People actually got a heart attack while they were not predicted to get one.
#10 people were predicted to get a heart attack while they did not get one.




accuracy = 33 + 23 / 33 + 23 + 10 + 10
#accuracy = 56 / 76
#accuracy = 0.73684210526

print(accuracy)

#With the new model that we just built, we have a higher accuracy to detect if a person will get a heart attack or not.


#You can try this out with a combination of different set of variables, or add more variables to this to see if that improves the accuracy of the model?


