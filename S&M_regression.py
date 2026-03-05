
#Simple Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("YOUR_FILE_PATH/YOUR_FILE.csv")  
# 🔴 CHANGE FILE PATH

X = df[["FEATURE_COLUMN"]]  
# 🔴 CHANGE FEATURE_COLUMN

y = df["TARGET_COLUMN"]  
# 🔴 CHANGE TARGET_COLUMN

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


#Multiple Linear Regression

X = df.drop("TARGET_COLUMN", axis=1)  
# 🔴 CHANGE TARGET_COLUMN

y = df["TARGET_COLUMN"]  
# 🔴 CHANGE TARGET_COLUMN

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)