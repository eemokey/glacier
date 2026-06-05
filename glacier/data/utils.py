import numpy as np 
import os 
import pandas as pd
import os 
import joblib 
import torch


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a specified file path into a pandas DataFrame,
    handling various file extensions.

    Args:
        data_path (str): The full path to the data file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the file extension is unsupported.
        Exception: For other potential pandas reading errors.
    """
    # 1. Check if the file exists 
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: The file was not found at '{data_path}'")


    # 2. Get the file extension 
    _, file_extension = os.path.splitext(data_path)
    file_extension = file_extension.lower()

    # 3. Load the data based on the extension 
    print(f"Attempting to load file with extension: '{file_extension}'...{data_path}")

    try:
        if file_extension == '.csv':
            return pd.read_csv(data_path)
        
        elif file_extension == '.txt':
            return pd.read_csv(data_path, delimiter=r'\s+') # Handles various whitespace

        elif file_extension in ['.xls', '.xlsx']:
            # Use read_excel for Excel files
            return pd.read_excel(data_path)

        elif file_extension == '.json':
            # Use read_json for JSON files
            return pd.read_json(data_path)
        
        elif file_extension == '.parquet':
            # Use read_parquet for Parquet files
            return pd.read_parquet(data_path)
        
        elif file_extension == '.joblib':
            # Use read_parquet for Parquet files
            return joblib.load(data_path)
        
        elif file_extension == '.tab':
            return pd.read_csv(data_path, sep='\t')
        
        elif file_extension == '.pt': # for gin graphs 
            return torch.load(data_path)
        
        elif file_extension == '.npz':
            return np.load(data_path, allow_pickle=True)
        
        else:
            # If the extension is not supported, raise an error
            raise ValueError(f"Unsupported file extension: '{file_extension}'. "
                             "Please use one of: .csv, .txt, .xlsx, .xls, .json, .parquet")
    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        # Re-raise the exception after printing the message
        raise




def load_rdkit(filename):
    """
    Returns loaded rdkit data

    Args:
        filename (str): filename and path to load OR
        chunk_id (int or None): Optional chunk ID to load specific file slice (e.g. data_chunk_0.csv)
        num_chunks: number of chunks

    NOTE:
    rdkit data saved as array of one sample of a dictionary of 200 features 
        with feature1: feature1 value, feature2:feature2value, for every sample.
        
    
    Returns:
        array of rdkit descriptors 
    """

    mydata = filename

    # LOAD data 
    rdkit_dict = load_data(mydata)['arr']

    # Convert to pandas df for easy handling 
    rdkit_df = pd.DataFrame(list(rdkit_dict))

    # Extract purely numerical values
    feature_matrix = rdkit_df.to_numpy()
    feature_names = rdkit_df.columns.to_numpy()

    # Handle NaNs (Crucial)
    if np.isnan(feature_matrix).any():
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        print('NANS found in rdkit, setting them to 0')

    # Apply Log Scaling (Crucial for stability)
    feature_matrix = np.sign(feature_matrix) * np.log1p(np.abs(feature_matrix))

    # Clip extremems (Optional safety)
    feature_matrix = np.clip(feature_matrix, -10.0, 10.0)


    return feature_matrix
