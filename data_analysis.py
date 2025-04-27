# VIBUSHAN S                   (24MSRCI010)
# KHADELA ASHISH MAHESHBHAI    (24MSRCI012)
# MUNIWALA KAUSHAL VIRENDRA KR (24MSRCI013)

# 3	
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