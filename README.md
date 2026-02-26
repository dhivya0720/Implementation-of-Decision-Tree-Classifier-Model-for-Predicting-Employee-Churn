# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DHIVYA DARSHNEE U
RegisterNumber:  212225220027

import pandas as pd

data = pd.read_csv("Employee.csv")


data.columns = data.columns.str.strip()

print("data.head():")
print(data.head())

print("data.info():")
print(data.info())

print("isnull() and sum():")
print(data.isnull().sum())

print("data value counts():")
print(data["left"].value_counts())


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

print("Encoding categorical columns:")


data["salary"] = le.fit_transform(data["salary"])
data["Departments"] = le.fit_transform(data["Departments"])

print(data.head())


print("x.head():")


x = data[["satisfaction_level",
          "last_evaluation",
          "number_project",
          "average_montly_hours",
          "time_spend_company",
          "Work_accident",
          "promotion_last_5years",
          "Departments",
          "salary"]]

print(x.head())

y = data["left"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")

print(x_train.dtypes)

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)


print("Accuracy value:")
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)


print("Data Prediction:")


print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 0, 2, 1]]))


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plot_tree(dt,
          feature_names=x.columns,
          class_names=['Stayed', 'Left'],
          filled=True)

plt.show()
*/
```

## Output:
<img width="872" height="305" alt="image" src="https://github.com/user-attachments/assets/8745a8f7-06e8-4870-8ac8-131eb25f6fc3" />

<img width="610" height="223" alt="image" src="https://github.com/user-attachments/assets/cf171c2b-ddca-48d3-808a-4781523764e2" />

<img width="613" height="340" alt="image" src="https://github.com/user-attachments/assets/50a2c24b-6c9d-4b4d-8e7b-e9f13056b6b3" />

<img width="633" height="416" alt="image" src="https://github.com/user-attachments/assets/db06fbbd-77c7-4780-9d52-17fc9bd59ec8" />

<img width="850" height="392" alt="image" src="https://github.com/user-attachments/assets/b5d000c3-79e7-4b79-b260-96b093640773" />

<img width="988" height="368" alt="image" src="https://github.com/user-attachments/assets/a250170e-9591-4a09-8d7f-d1fc85f60871" />

<img width="622" height="386" alt="image" src="https://github.com/user-attachments/assets/19e98869-c9aa-499e-b4c7-1fa1bd349b11" />

<img width="956" height="636" alt="image" src="https://github.com/user-attachments/assets/8be57e5e-828a-4daf-abd2-e3b1b655ffd5" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
