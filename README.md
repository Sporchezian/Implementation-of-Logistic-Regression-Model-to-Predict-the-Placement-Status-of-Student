# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1) Load and preprocess data: Read the dataset, drop the salary column, and convert categorical variables to numeric using one‑hot encoding.

2) Split dataset: Separate features (X) and target (y), then divide into training and testing sets.

3) Train and evaluate model: Fit a logistic regression model on the training data and compute accuracy on the test set.

4) Visualize logistic curve: Select one feature, train a single‑feature logistic regression, plot the scatter of data points, and overlay the sigmoid probability     curve.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: PORCHEZIAN S 

RegisterNumber: 212225040304

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("Placement_Data.csv")

# Drop salary column (contains missing values)
data = data.drop("salary", axis=1)

# Convert categorical data to numeric
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))


# -------------------------------
# 📈 Logistic Regression Plot
# -------------------------------

# Use only ONE feature for plotting
X1 = X.iloc[:, 0].values.reshape(-1, 1)

# Train model on single feature
model_plot = LogisticRegression(max_iter=1000)
model_plot.fit(X1, y)

# Scatter plot
plt.scatter(X1, y, color='blue')

# Sigmoid curve
x_values = np.linspace(X1.min(), X1.max(), 100)
y_values = model_plot.predict_proba(x_values.reshape(-1,1))[:,1]

plt.plot(x_values, y_values)

plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression Curve")
plt.show()

```

## Output:

<img width="783" height="618" alt="image" src="https://github.com/user-attachments/assets/86879a6c-2837-43e1-8aa7-c73b6af03542" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
