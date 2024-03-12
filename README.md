# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import numpy as np 3.Plot the points
3. intiLiaze thhe program
4. End the program



## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: M.Harini
RegisterNumber:  212222240035
*/
```
```
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #Calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #Update theto using gradient descent
        theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
#Learn model Parameters
theta= linear_regression(X1_Scaled,Y1_Scaled)
#Predict data value for a new value point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/Harinimuthu17/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/130278614/3f8ffa4e-3b78-4469-aac0-dfd892464666)
![image](https://github.com/Harinimuthu17/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/130278614/815b76b4-06fc-4e24-b2fa-3acfb3717f25)
![image](https://github.com/Harinimuthu17/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/130278614/f2d7a0b0-4695-4857-987b-f3308de9d75a)
![image](https://github.com/Harinimuthu17/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/130278614/d1deccbd-76f8-489e-b317-ce3ce7ffedb0)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

