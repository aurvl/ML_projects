import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess(df, train):
    """
    This function handles missing and aberrant (outlier) values in a DataFrame by replacing them with the median of the respective column's distribution.
    It also applies scaling to standardize the data.

    Args:
        df (DataFrame): The dataset containing columns with missing or aberrant values.
        train (bool): A flag indicating whether the function is being applied to a training set (True) or a test set (False).
                      If True, the function will fit and save an imputer, scaler, and outlier bounds; if False, it will load and apply the saved models and bounds.

    Returns:
        DataFrame: A transformed DataFrame with missing and aberrant values replaced by the median, and features standardized.
    
    Process:
        1. Outlier Detection and Replacement: Outliers are identified using the IQR (Interquartile Range) method.
           Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are considered outliers and replaced with NaN.
        2. Imputation: Missing values (including outliers replaced by NaN) are filled with the median of each column.
           If train is True, an imputer is fit and saved for later use. If train is False, the previously saved imputer is loaded.
        3. Scaling: Standardizes the data to have a mean of 0 and a standard deviation of 1.
           If train is True, a scaler is fit and saved; otherwise, the saved scaler is loaded and applied.
    """
    
    # Step 1: Handle outliers based on the IQR method
    if train:
        # Calculate and save outlier bounds for each column in the training data
        outlier_bounds = {}
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_bounds[col] = (lower_bound, upper_bound)
            
            # Replace outliers with NaN in training data
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
        
        # Save the outlier bounds
        with open('outlier_bounds.pickle', 'wb') as f:
            pickle.dump(outlier_bounds, f)
    else:
        # Load and apply outlier bounds in the test data
        with open('outlier_bounds.pickle', 'rb') as f:
            outlier_bounds = pickle.load(f)
        
        for col in df.columns:
            lower_bound, upper_bound = outlier_bounds[col]
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
    
    # Step 2: Impute missing values with the median
    if train:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df)
        with open('imputer.pickle', 'wb') as f:
            pickle.dump(imputer, f)
    else:
        with open('imputer.pickle', 'rb') as f:
            imputer = pickle.load(f)
    
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    
    # Step 3: Standardize the data
    if train:
        scaler = StandardScaler()
        scaler.fit(df)
        with open('scaler.pickle', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open('scaler.pickle', 'rb') as f:
            scaler = pickle.load(f)
    
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df