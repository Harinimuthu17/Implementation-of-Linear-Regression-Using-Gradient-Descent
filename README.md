# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1. Use the standard libraries in python for Gradient Design.
 2. Upload the dataset and check any null value using .isnull() function.
 3. Declare the default values for linear regression.
 4. Calculate the loss usinng Mean Square Error.
 5. Predict the value of y.
 6. Plot the graph respect to hours and scores using scatter plot function.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:M.Harini
RegisterNumber:  212222240035
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("student_scores.csv")
data.head()
data.isnull().sum()
x = data.Hours
x.head()
y = data.Scores
y.head()
n = len(x)
m = 0
c = 0
L = 0.001
loss = []
for i in range(10000):
    ypred = m*x + c
    MSE = (1/n) * sum((ypred - y)*2)
    dm = (2/n) * sum(x*(ypred-y))
    dc = (2/n) * sum(ypred-y)
    c = c-L*dc
    m = m-L*dm
    loss.append(MSE)
    #print(m)
print(m,c)
y_pred = m*x + c
plt.scatter(x,y,color = "pink")
plt.plot(x,y_pred)
plt.xlabel("Study hours")
plt.ylabel("Scores")
plt.title("Study hours vs. Scores")
plt.plot(loss)
plt.xlabel("Iterations")
plt.ylabel("loss")
*/
```

## Output:
![image](https://user-images.githubusercontent.com/119389139/230386308-e04bfb79-b231-453b-953e-e45512f79148.png)

![image](https://user-images.githubusercontent.com/119389139/230386410-d4ccb116-c4d8-4c4b-b348-f5ccba787338.png)

![image](https://user-images.githubusercontent.com/119389139/230386510-63de0d84-f31d-4a1c-a9fd-4972f86cf64e.png)

![image](https://user-images.githubusercontent.com/119389139/230386833-3d102068-46b6-479f-83c7-cb87f732526e.png)

![image](https://user-images.githubusercontent.com/119389139/230389941-e78316c2-0ef7-40aa-8a98-036822924a2b.png)

![image](https://user-images.githubusercontent.com/119389139/230390024-44c07657-bcc7-42d7-a710-b70cc6d5917b.png)

![image](https://user-images.githubusercontent.com/119389139/230390104-41a9a384-10fe-4380-ba90-3d133ea40167.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
