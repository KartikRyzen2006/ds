# Simple Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\dataset\iris-write-from-docker.csv")  
# CHANGE FILE PATH if required

X = df[["sepal_length"]]  
# CHANGE FEATURE COLUMN if needed

y = df["sepal_width"]  
# CHANGE TARGET COLUMN if needed

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


#Multiple Linear Regression

# Remove categorical column
df_numeric = df.drop(columns=["class"])  
# REMOVE OR CHANGE COLUMN if dataset contains other text columns

X = df_numeric.drop("sepal_width", axis=1)  
# CHANGE TARGET COLUMN

y = df_numeric["sepal_width"]  
# CHANGE TARGET COLUMN

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
