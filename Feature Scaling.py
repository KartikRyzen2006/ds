#Feature Scaling
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/dataset/iris-write-from-docker.csv")  
# 🔴 CHANGE FILE PATH

numeric_cols = df.select_dtypes(include='number').columns

minmax = MinMaxScaler()
df[numeric_cols] = minmax.fit_transform(df[numeric_cols])

print("After MinMax Scaling:")
print(df.head())

std = StandardScaler()
df[numeric_cols] = std.fit_transform(df[numeric_cols])

print("After Standard Scaling:")
print(df.head())

#Feature Dummification / Encoding
le = LabelEncoder()
df["Encoded"] = le.fit_transform(df["class"])

print(df.head())
