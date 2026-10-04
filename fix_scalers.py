"""
Run this once, in the same folder as creditcard.csv, to fix the scaler bug.
It fits two SEPARATE scalers instead of reusing one object for both columns.
"""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("creditcard.csv")

scaler_amount = StandardScaler().fit(df[["Amount"]])
scaler_time = StandardScaler().fit(df[["Time"]])

joblib.dump(scaler_amount, "scaler_amount.pkl")
joblib.dump(scaler_time, "scaler_time.pkl")

print("scaler_amount -> mean:", scaler_amount.mean_, " scale:", scaler_amount.scale_)
print("scaler_time   -> mean:", scaler_time.mean_, " scale:", scaler_time.scale_)
print("\nSaved scaler_amount.pkl and scaler_time.pkl - both correctly fit.")
