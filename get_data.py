from sklearn.datasets import fetch_openml
creditcard = fetch_openml(data_id=1597, as_frame=True)
df = creditcard.frame
print(df.columns)
print(df.shape)
