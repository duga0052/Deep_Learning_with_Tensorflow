import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file is not found.
        pd.errors.EmptyDataError: If the file is empty.
        Exception: For any other error during data loading.
    """
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        
         # Display the first few rows of the dataframe
        print("First few rows of the dataframe:")
        print(df.head())

        print("\nDataframe information:")
        # Display information about the dataframe
        df.info()
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """
    Preprocess the data for model training.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: x_train, x_test, y_train, y_test

    Raises:
        ValueError: If 'Attrition' column is not found in the DataFrame.
        Exception: For any other error during preprocessing.
    """
    try:
        logger.info("Starting data preprocessing")
        if 'Attrition' not in df.columns:
            raise ValueError("'Attrition' column not found in the DataFrame")

        Y = df.Attrition
        X = df.drop(columns=['Attrition'])
        
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
        
        logger.info(f"Preprocessing complete. Train shape: {x_train.shape}, Test shape: {x_test.shape}")
        return x_train, x_test, y_train, y_test
    except ValueError as ve:
        logger.error(f"ValueError during preprocessing: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise