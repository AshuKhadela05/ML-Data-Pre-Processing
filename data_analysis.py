# a) Creation and Loading different types of datasets in Python using the required libraries.
#    i.	Creation using pandas
#   ii.	Loading CSV dataset files using Pandas
#  iii.	Loading datasets using sklearn
# b) Write a python program to compute Mean, Median, Mode, Variance, Standard Deviation using Datasets
# c) Demonstrate various data pre-processing techniques for a given dataset.
#    Write a python program to compute
#    i.	Reshaping the data,
#   ii.	Filtering the data,
#  iii.	Merging the data
#   iv.	Handling the missing values in datasets
# Feature Normalization: Min-max normalization	


# Importing all required libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as scipy_stats
import os

# ==============================================
# 1. DATASET CREATION AND LOADING
# ==============================================

def create_dataset():
    """Create a sample dataset using pandas"""
    data = {
        'Student_ID': [101, 102, 103, 104, 105],
        'Name': ['Ashish', 'Kaushal', 'Vibushan', 'Sujal', 'Kush'],
        'Age': [20, 21, 19, 22, 20],
        'Math_Score': [85, 78, 92, 88, 90],
        'Science_Score': [90, 85, 88, 92, 85],
        'Graduated': [True, False, True, True, False]
    }
    df = pd.DataFrame(data)
    print("\nCreated Dataset:")
    print(df)
    return df

def load_csv_data():
    """Load CSV data with multiple fallback options"""
    urls = [
        # Primary URL
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        # Mirror URL
        "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    ]
    
    local_path = "iris.csv"  # Try local file if exists
    
    # Try multiple URLs
    for url in urls:
        try:
            df = pd.read_csv(url)
            print(f"\nSuccessfully loaded dataset from URL: {url}")
            print("First 5 rows:")
            print(df.head())
            return df
        except Exception as e:
            print(f"Failed to load from URL {url}: {str(e)[:100]}...")
    
    # Try local file
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            print("\nLoaded dataset from local file iris.csv")
            print("First 5 rows:")
            print(df.head())
            return df
        except Exception as e:
            print(f"Failed to load local file: {e}")
    
    # Fallback to sklearn dataset
    print("\nFalling back to sklearn iris dataset")
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])
    print("First 5 rows:")
    print(df.head())
    return df

def load_sklearn_data():
    """Load dataset using sklearn"""
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])
    print("\nLoaded sklearn Dataset (First 5 rows):")
    print(iris_df.head())
    return iris_df

# ==============================================
# 2. STATISTICAL COMPUTATIONS
# ==============================================

def compute_statistics(df, column_name):
    """Compute basic statistics for a dataframe column"""
    try:
        data = df[column_name]
        statistics = {
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Mode': scipy_stats.mode(data, keepdims=True)[0][0],
            'Variance': np.var(data),
            'Standard Deviation': np.std(data)
        }
        print(f"\nStatistics for '{column_name}':")
        for key, value in statistics.items():
            print(f"{key}: {value:.4f}")
        return statistics
    except Exception as e:
        print(f"\nError computing statistics for column '{column_name}': {e}")
        return None

# ==============================================
# 3. DATA PRE-PROCESSING TECHNIQUES
# ==============================================

def reshape_data(df):
    """Demonstrate reshaping operations"""
    print("\nOriginal DataFrame:")
    print(df.head())
    
    melted = pd.melt(df, id_vars=['Student_ID', 'Name'], 
                    value_vars=['Math_Score', 'Science_Score'],
                    var_name='Subject', value_name='Score')
    print("\nMelted DataFrame (long format):")
    print(melted.head())
    
    pivoted = melted.pivot(index=['Student_ID', 'Name'], 
                          columns='Subject', values='Score').reset_index()
    print("\nPivoted DataFrame (back to wide format):")
    print(pivoted.head())
    
    return melted, pivoted

def filter_data(df):
    """Demonstrate filtering operations"""
    print("\nOriginal DataFrame:")
    print(df.head())
    
    filtered_age = df[df['Age'] > 20]
    print("\nStudents older than 20:")
    print(filtered_age)
    
    filtered_scores = df[(df['Math_Score'] > 85) & (df['Science_Score'] > 85)]
    print("\nStudents with scores > 85 in both subjects:")
    print(filtered_scores)
    
    selected_cols = df[['Name', 'Math_Score']]
    print("\nSelected columns (Name and Math_Score):")
    print(selected_cols.head())
    
    return filtered_age, filtered_scores, selected_cols

def merge_data(df):
    """Demonstrate merging operations"""
    df1 = df[['Student_ID', 'Name', 'Age']]
    df2 = df[['Student_ID', 'Math_Score', 'Science_Score']]
    
    print("\nFirst DataFrame to merge:")
    print(df1.head())
    print("\nSecond DataFrame to merge:")
    print(df2.head())
    
    merged = pd.merge(df1, df2, on='Student_ID')
    print("\nMerged DataFrame:")
    print(merged.head())
    
    return merged

def handle_missing_values():
    """Demonstrate missing value handling"""
    data = {
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)
    
    print("\nDataFrame with missing values:")
    print(df)
    
    print("\nMissing values count:")
    print(df.isna().sum())
    
    dropped = df.dropna()
    print("\nAfter dropping rows with missing values:")
    print(dropped)
    
    filled = df.fillna(df.mean())
    print("\nAfter filling missing values with mean:")
    print(filled)
    
    return df, dropped, filled

def normalize_data(df, columns):
    """Perform min-max normalization"""
    print(f"\nOriginal Data (columns: {columns}):")
    print(df[columns].head())
    
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df[columns])
    
    print("\nAfter Min-Max Normalization:")
    print(df_normalized[columns].head())
    
    return df_normalized

# ==============================================
# MAIN EXECUTION
# ==============================================

def main():
    print("="*50)
    print("DATASET CREATION AND LOADING DEMONSTRATION")
    print("="*50)
    
    created_df = create_dataset()
    csv_df = load_csv_data()
    sklearn_df = load_sklearn_data()
    
    print("\n" + "="*50)
    print("STATISTICAL COMPUTATIONS DEMONSTRATION")
    print("="*50)
    
    # Compute statistics on created dataset if CSV failed
    if csv_df is not None:
        compute_statistics(csv_df, 'sepal_length')
    else:
        print("\nUsing created dataset for statistics demonstration")
        compute_statistics(created_df, 'Math_Score')
    
    print("\n" + "="*50)
    print("DATA PRE-PROCESSING TECHNIQUES DEMONSTRATION")
    print("="*50)
    
    melted, pivoted = reshape_data(created_df)
    filtered_age, filtered_scores, selected_cols = filter_data(created_df)
    merged_df = merge_data(created_df)
    
    original_missing, dropped_missing, filled_missing = handle_missing_values()
    
    # Normalization demonstration
    if csv_df is not None:
        normalize_data(csv_df, ['sepal_length', 'sepal_width'])
    else:
        print("\nUsing created dataset for normalization demonstration")
        normalize_data(created_df, ['Math_Score', 'Science_Score'])
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()

# -----------------------------------------OUTPUT------------------------------------------------

==================================================
DATASET CREATION AND LOADING DEMONSTRATION
==================================================

Created Dataset:
   Student_ID      Name  Age  Math_Score  Science_Score  Graduated
0         101    Ashish   20          85             90       True
1         102   Kaushal   21          78             85      False
2         103  Vibushan   19          92             88       True
3         104     Sujal   22          88             92       True
4         105      Kush   20          90             85      False

Successfully loaded dataset from URL: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
First 5 rows:
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa

Loaded sklearn Dataset (First 5 rows):
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2     0.0
1                4.9               3.0                1.4               0.2     0.0
2                4.7               3.2                1.3               0.2     0.0
3                4.6               3.1                1.5               0.2     0.0
4                5.0               3.6                1.4               0.2     0.0

==================================================
STATISTICAL COMPUTATIONS DEMONSTRATION
==================================================

Statistics for 'sepal_length':
Mean: 5.8433
Median: 5.8000
Mode: 5.0000
Variance: 0.6811
Standard Deviation: 0.8253

==================================================
DATA PRE-PROCESSING TECHNIQUES DEMONSTRATION
==================================================

Original DataFrame:
   Student_ID      Name  Age  Math_Score  Science_Score  Graduated
0         101    Ashish   20          85             90       True
1         102   Kaushal   21          78             85      False
2         103  Vibushan   19          92             88       True
3         104     Sujal   22          88             92       True
4         105      Kush   20          90             85      False

Melted DataFrame (long format):
   Student_ID      Name     Subject  Score
0         101    Ashish  Math_Score     85
1         102   Kaushal  Math_Score     78
2         103  Vibushan  Math_Score     92
3         104     Sujal  Math_Score     88
4         105      Kush  Math_Score     90

Pivoted DataFrame (back to wide format):
Subject  Student_ID      Name  Math_Score  Science_Score
0               101    Ashish          85             90
1               102   Kaushal          78             85
2               103  Vibushan          92             88
3               104     Sujal          88             92
4               105      Kush          90             85

Original DataFrame:
   Student_ID      Name  Age  Math_Score  Science_Score  Graduated
0         101    Ashish   20          85             90       True
1         102   Kaushal   21          78             85      False
2         103  Vibushan   19          92             88       True
3         104     Sujal   22          88             92       True
4         105      Kush   20          90             85      False

Students older than 20:
   Student_ID     Name  Age  Math_Score  Science_Score  Graduated
1         102  Kaushal   21          78             85      False
3         104    Sujal   22          88             92       True

Students with scores > 85 in both subjects:
   Student_ID      Name  Age  Math_Score  Science_Score  Graduated
2         103  Vibushan   19          92             88       True
3         104     Sujal   22          88             92       True

Selected columns (Name and Math_Score):
       Name  Math_Score
0    Ashish          85
1   Kaushal          78
2  Vibushan          92
3     Sujal          88
4      Kush          90

First DataFrame to merge:
   Student_ID      Name  Age
0         101    Ashish   20
1         102   Kaushal   21
2         103  Vibushan   19
3         104     Sujal   22
4         105      Kush   20

Second DataFrame to merge:
   Student_ID  Math_Score  Science_Score
0         101          85             90
1         102          78             85
2         103          92             88
3         104          88             92
4         105          90             85

Merged DataFrame:
   Student_ID      Name  Age  Math_Score  Science_Score
0         101    Ashish   20          85             90
1         102   Kaushal   21          78             85
2         103  Vibushan   19          92             88
3         104     Sujal   22          88             92
4         105      Kush   20          90             85

DataFrame with missing values:
     A    B   C
0  1.0  5.0  10
1  2.0  NaN  20
2  NaN  NaN  30
3  4.0  8.0  40

Missing values count:
A    1
B    2
C    0
dtype: int64

After dropping rows with missing values:
     A    B   C
0  1.0  5.0  10
3  4.0  8.0  40

After filling missing values with mean:
          A    B   C
0  1.000000  5.0  10
1  2.000000  6.5  20
2  2.333333  6.5  30
3  4.000000  8.0  40

Original Data (columns: ['sepal_length', 'sepal_width']):
   sepal_length  sepal_width
0           5.1          3.5
1           4.9          3.0
2           4.7          3.2
3           4.6          3.1
4           5.0          3.6

After Min-Max Normalization:
   sepal_length  sepal_width
0      0.222222     0.625000
1      0.166667     0.416667
2      0.111111     0.500000
3      0.083333     0.458333
4      0.194444     0.666667

All operations completed successfully!
