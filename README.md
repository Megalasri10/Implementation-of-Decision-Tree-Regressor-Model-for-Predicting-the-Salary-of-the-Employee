# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Get the independent variable X and dependent variable Y.

2.Calculate the mean of the X -values and the mean of the Y -values.

3.Find the slope m of the line of best fit using the formula.
<img width="296" height="134" alt="Screenshot 2026-04-24 142329" src="https://github.com/user-attachments/assets/12a974c4-1f2b-448b-a008-ab32569d625b" />


4. Compute the y -intercept of the line by using the formula:

<img width="209" height="51" alt="Screenshot 2026-04-24 142402" src="https://github.com/user-attachments/assets/ea10d804-b715-47bc-9725-ec6901f76653" />

5. Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Megala M S
*/
RegisterNumber:  212225040230
# Implementation of Decision Tree Regressor Model
# Predicting Employee Salary

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Step 2: Create Dataset
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
    "Salary": [25000, 30000, 35000, 45000, 50000,
               60000, 65000, 75000, 85000, 95000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display Dataset
print("Dataset:\n")
print(df)

# Step 3: Split Features and Target
X = df[["Experience", "Age"]]
y = df["Salary"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Decision Tree Regressor
model = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Evaluation:")

print("Mean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

# Step 8: Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance)

# Step 9: Visualize Decision Tree
plt.figure(figsize=(14,8))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()

# Step 10: Visualization of Actual vs Predicted
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual Salary vs Predicted Salary")
plt.grid(True)

plt.show()

# Step 11: Custom Prediction
employee_data = [[5, 30]]

predicted_salary = model.predict(employee_data)

print("\nPredicted Salary for Employee:")
print(f"Predicted Salary = {predicted_salary[0]:.2f}") 

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

<img width="1322" height="660" alt="image" src="https://github.com/user-attachments/assets/710863b6-da8c-482f-b244-e0797a0aba5e" />
<img width="1346" height="685" alt="image" src="https://github.com/user-attachments/assets/30b3da18-c607-4e28-91f2-47c748c87df4" />
<img width="1327" height="665" alt="image" src="https://github.com/user-attachments/assets/173068b4-bf1c-4571-8c4e-00aacbb7b234" />
<img width="635" height="135" alt="image" src="https://github.com/user-attachments/assets/7edd1250-bfa6-4577-8e37-0f264f8025a9" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
