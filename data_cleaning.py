import pandas as pd
import chardet

file_path = "customer_data.csv"
encoding = 'Big5'



df = pd.read_csv(file_path, encoding=encoding)
print(f"Successfully read '{file_path}' with encoding '{encoding}'")


print(df.describe())