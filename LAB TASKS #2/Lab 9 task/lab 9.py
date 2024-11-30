import pandas as pd


dataset_path = "All_Diets.csv"  
data = pd.read_csv(dataset_path)

print("First 5 Rows of the Dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDescriptive Statistics:")
print(data.describe())
