"""Transaction preprocessing pipeline for quantum circuit encoding.

This module implements the TransactionPreprocessor class that transforms
raw transaction data into normalized feature vectors suitable for quantum
angle encoding.
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os


class TransactionPreprocessor:
    """Handles end-to-end preprocessing of transaction data.
    
    This preprocessor transforms raw transaction data into normalized feature
    vectors suitable for quantum angle encoding. Features are scaled to the
    range [0, π] for use in quantum rotation gates.
    
    Attributes:
        categorical_columns: List of categorical feature names
        numerical_columns: List of numerical feature names
        scaler: MinMaxScaler for scaling features to [0, π]
        label_encoders: Dictionary mapping column names to fitted LabelEncoders
        is_fitted: Flag indicating whether the preprocessor has been fitted
    """
    
    def __init__(self, categorical_columns: List[str], numerical_columns: List[str]):
        """Initialize preprocessor with column specifications.
        
        Args:
            categorical_columns: List of categorical feature names
            numerical_columns: List of numerical feature names
        """
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fit preprocessing transformations on training data.
        
        This method:
        1. Fits missing value imputers for numerical and categorical columns
        2. Fits LabelEncoders for each categorical column
        3. Fits MinMaxScaler on the combined feature matrix
        4. Sets is_fitted flag to True
        
        Args:
            df: DataFrame containing raw transaction data
            
        Raises:
            ValueError: If DataFrame is missing required columns
        """
        # Validate required columns are present
        missing_cols = []
        for col in self.categorical_columns + self.numerical_columns:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Step 1: Fit and transform missing value imputers
        # Handle numerical columns
        if self.numerical_columns:
            numerical_data = df[self.numerical_columns].values
            self.numerical_imputer.fit(numerical_data)
            numerical_imputed = self.numerical_imputer.transform(numerical_data)
        else:
            numerical_imputed = np.array([]).reshape(len(df), 0)
        
        # Handle categorical columns
        if self.categorical_columns:
            categorical_data = df[self.categorical_columns].values
            self.categorical_imputer.fit(categorical_data)
            categorical_imputed = self.categorical_imputer.transform(categorical_data)
            
            # Step 2: Fit LabelEncoder for each categorical column
            encoded_categorical = []
            for idx, col in enumerate(self.categorical_columns):
                encoder = LabelEncoder()
                # Fit on imputed data for this column
                encoded_values = encoder.fit_transform(categorical_imputed[:, idx])
                self.label_encoders[col] = encoder
                encoded_categorical.append(encoded_values.reshape(-1, 1))
            
            # Concatenate all encoded categorical features
            encoded_categorical_array = np.hstack(encoded_categorical)
        else:
            encoded_categorical_array = np.array([]).reshape(len(df), 0)
        
        # Step 3: Concatenate encoded categorical and numerical features
        combined_features = np.hstack([encoded_categorical_array, numerical_imputed])
        
        # Step 4: Fit MinMaxScaler on combined feature matrix
        self.scaler.fit(combined_features)
        
        # Step 5: Set is_fitted to True
        self.is_fitted = True
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformations to generate feature vectors.
        
        This method applies the preprocessing pipeline fitted during training
        to new data for inference. It performs the same transformations in the
        same order: imputation, label encoding, and scaling.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Array of shape (n_samples, n_features) with values in [0, π]
            
        Raises:
            ValueError: If transform is called before fit
        """
        # Step 1: Check is_fitted flag and raise ValueError if False
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        # Step 2: Apply imputation to handle missing values
        # Handle numerical columns
        if self.numerical_columns:
            numerical_data = df[self.numerical_columns].values
            numerical_imputed = self.numerical_imputer.transform(numerical_data)
        else:
            numerical_imputed = np.array([]).reshape(len(df), 0)
        
        # Handle categorical columns
        if self.categorical_columns:
            categorical_data = df[self.categorical_columns].values
            categorical_imputed = self.categorical_imputer.transform(categorical_data)
            
            # Step 3: Apply label encoding to categorical features
            encoded_categorical = []
            for idx, col in enumerate(self.categorical_columns):
                encoder = self.label_encoders[col]
                # Transform imputed data for this column
                encoded_values = encoder.transform(categorical_imputed[:, idx])
                encoded_categorical.append(encoded_values.reshape(-1, 1))
            
            # Concatenate all encoded categorical features
            encoded_categorical_array = np.hstack(encoded_categorical)
        else:
            encoded_categorical_array = np.array([]).reshape(len(df), 0)
        
        # Concatenate encoded categorical and numerical features
        combined_features = np.hstack([encoded_categorical_array, numerical_imputed])
        
        # Step 4: Apply MinMaxScaler transformation to all features
        scaled_features = self.scaler.transform(combined_features)
        
        # Step 5: Return numpy array with shape (n_samples, n_features) and values in [0, π]
        return scaled_features
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience method combining fit and transform.
        
        This method first fits the preprocessing pipeline on the provided data
        and then applies the transformations to generate feature vectors.
        
        Args:
            df: DataFrame containing raw transaction data
            
        Returns:
            Array of shape (n_samples, n_features) with values in [0, π]
        """
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, features: np.ndarray) -> pd.DataFrame:
        """Reverse transformation for debugging and interpretation.
        
        This method reverses the preprocessing pipeline to convert normalized
        feature vectors back to original-scale values. It performs inverse
        scaling followed by inverse label encoding for categorical features.
        
        Note: This is primarily for debugging and interpretation. Due to
        imputation and encoding, the inverse transform may not exactly
        recover the original raw data.
        
        Args:
            features: Normalized feature array of shape (n_samples, n_features)
                     with values in [0, π]
            
        Returns:
            DataFrame with original-scale values
            
        Raises:
            ValueError: If inverse_transform is called before fit
        """
        # Check if the preprocessor has been fitted
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform. Call fit() first.")
        
        # Step 1: Inverse scale using MinMaxScaler
        unscaled_features = self.scaler.inverse_transform(features)
        
        # Step 2: Split back into categorical and numerical features
        n_categorical = len(self.categorical_columns)
        n_numerical = len(self.numerical_columns)
        
        if n_categorical > 0:
            encoded_categorical = unscaled_features[:, :n_categorical]
        else:
            encoded_categorical = np.array([]).reshape(len(features), 0)
        
        if n_numerical > 0:
            numerical_features = unscaled_features[:, n_categorical:]
        else:
            numerical_features = np.array([]).reshape(len(features), 0)
        
        # Step 3: Inverse label encoding for categorical features
        result_dict = {}
        
        for idx, col in enumerate(self.categorical_columns):
            encoder = self.label_encoders[col]
            # Round to nearest integer since encoded values should be integers
            encoded_values = np.round(encoded_categorical[:, idx]).astype(int)
            # Clip values to valid range for the encoder
            encoded_values = np.clip(encoded_values, 0, len(encoder.classes_) - 1)
            # Inverse transform to get original categorical values
            decoded_values = encoder.inverse_transform(encoded_values)
            result_dict[col] = decoded_values
        
        # Step 4: Add numerical features to result
        for idx, col in enumerate(self.numerical_columns):
            result_dict[col] = numerical_features[:, idx]
        
        # Step 5: Create and return DataFrame with original column order
        # Preserve original column order (categorical first, then numerical)
        ordered_columns = self.categorical_columns + self.numerical_columns
        return pd.DataFrame(result_dict)[ordered_columns]
    
    def save(self, filepath: str) -> None:
        """Save preprocessor state to disk for later restoration.
        
        This method serializes the complete preprocessor state including:
        - Fitted MinMaxScaler
        - Fitted LabelEncoders for each categorical column
        - Numerical and categorical imputers
        - Column specifications
        - is_fitted flag
        
        The state is saved using joblib for efficient serialization of
        scikit-learn objects.
        
        Args:
            filepath: Path where preprocessor state should be saved.
                     Will create parent directories if they don't exist.
        
        Raises:
            ValueError: If save is called before fit (when is_fitted is False)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor. Call fit() before save().")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Package all state into a dictionary
        state = {
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'is_fitted': self.is_fitted
        }
        
        # Serialize to disk using joblib
        joblib.dump(state, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'TransactionPreprocessor':
        """Load preprocessor state from disk.
        
        This class method restores a complete TransactionPreprocessor instance
        from a previously saved state file. All fitted transformations including
        scalers, encoders, and imputers are restored, allowing immediate use
        for inference without requiring retraining.
        
        Args:
            filepath: Path to saved preprocessor state file
        
        Returns:
            TransactionPreprocessor instance with restored state
        
        Raises:
            FileNotFoundError: If filepath does not exist
            ValueError: If loaded file is not a valid preprocessor state
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor state file not found: {filepath}")
        
        # Load state from disk
        try:
            state = joblib.load(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load preprocessor state: {str(e)}")
        
        # Validate state dictionary
        required_keys = [
            'categorical_columns', 'numerical_columns', 'scaler', 
            'label_encoders', 'numerical_imputer', 'categorical_imputer', 
            'is_fitted'
        ]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise ValueError(f"Invalid preprocessor state file. Missing keys: {missing_keys}")
        
        # Create new instance
        instance = cls(
            categorical_columns=state['categorical_columns'],
            numerical_columns=state['numerical_columns']
        )
        
        # Restore fitted state
        instance.scaler = state['scaler']
        instance.label_encoders = state['label_encoders']
        instance.numerical_imputer = state['numerical_imputer']
        instance.categorical_imputer = state['categorical_imputer']
        instance.is_fitted = state['is_fitted']
        
        return instance
