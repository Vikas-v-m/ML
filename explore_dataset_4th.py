import pandas as pd

def explore_dataset(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv("/content/Iris.csv")
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel("/content/Iris.xlsx")
    else:
        print("Unsupported file format. Please provide a CSV or Excel file.")
        return

    print("Dataset information:")
    print(df.info())

    print("\nFirst few rows of the dataset:")
    print(df.head())

    print("\nSummary statistics:")
    print(df.describe())

    print("\nUnique values for categorical columns:")
    for column in df.select_dtypes(include='object').columns:
        print(f"{column}: {df[column].unique()}")

file_path = 'Iris.csv'
explore_dataset("/content/Iris.csv")
