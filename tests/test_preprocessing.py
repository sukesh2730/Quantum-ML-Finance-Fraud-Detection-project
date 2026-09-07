"""Unit tests for TransactionPreprocessor."""

import pytest
import numpy as np
import pandas as pd
from quantum_fraud_detector.preprocessing.preprocessor import TransactionPreprocessor


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    return pd.DataFrame({
        'merchant_category': ['online_retail', 'grocery', 'gas_station', 'online_retail', 'grocery'],
        'card_type': ['visa', 'mastercard', 'visa', 'amex', 'visa'],
        'amount': [100.0, 50.0, 30.0, 200.0, 75.0],
        'time_of_day': [14.5, 9.0, 18.5, 22.0, 12.0],
        'location_distance': [5.0, 2.0, 10.0, 50.0, 3.0]
    })


@pytest.fixture
def preprocessor():
    """Create a TransactionPreprocessor instance."""
    categorical_cols = ['merchant_category', 'card_type']
    numerical_cols = ['amount', 'time_of_day', 'location_distance']
    return TransactionPreprocessor(categorical_cols, numerical_cols)


class TestFitTransform:
    """Tests for fit_transform method."""
    
    def test_fit_transform_basic_functionality(self, preprocessor, sample_transaction_data):
        """Test that fit_transform works and returns correct shape."""
        features = preprocessor.fit_transform(sample_transaction_data)
        
        # Check that result is a numpy array
        assert isinstance(features, np.ndarray)
        
        # Check shape: 5 samples, 5 features (2 categorical + 3 numerical)
        assert features.shape == (5, 5)
        
        # Check that preprocessor is fitted
        assert preprocessor.is_fitted is True
    
    def test_fit_transform_values_in_valid_range(self, preprocessor, sample_transaction_data):
        """Test that fit_transform produces values in [0, π] range."""
        features = preprocessor.fit_transform(sample_transaction_data)
        
        # All values should be in [0, π]
        assert np.all(features >= 0)
        assert np.all(features <= np.pi)
    
    def test_fit_transform_equivalent_to_fit_then_transform(self, preprocessor, sample_transaction_data):
        """Test that fit_transform is equivalent to calling fit then transform."""
        # Create two separate preprocessor instances with same config
        preprocessor2 = TransactionPreprocessor(
            preprocessor.categorical_columns,
            preprocessor.numerical_columns
        )
        
        # Method 1: fit_transform
        features1 = preprocessor.fit_transform(sample_transaction_data)
        
        # Method 2: fit then transform
        preprocessor2.fit(sample_transaction_data)
        features2 = preprocessor2.transform(sample_transaction_data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(features1, features2)
    
    def test_fit_transform_with_missing_values(self, preprocessor):
        """Test fit_transform handles missing values correctly."""
        data_with_missing = pd.DataFrame({
            'merchant_category': ['online_retail', None, 'gas_station', 'online_retail', 'grocery'],
            'card_type': ['visa', 'mastercard', None, 'amex', 'visa'],
            'amount': [100.0, np.nan, 30.0, 200.0, 75.0],
            'time_of_day': [14.5, 9.0, np.nan, 22.0, 12.0],
            'location_distance': [5.0, 2.0, 10.0, np.nan, 3.0]
        })
        
        features = preprocessor.fit_transform(data_with_missing)
        
        # Should not contain NaN values after imputation
        assert not np.any(np.isnan(features))
        
        # Should have correct shape
        assert features.shape == (5, 5)


class TestInverseTransform:
    """Tests for inverse_transform method."""
    
    def test_inverse_transform_basic_functionality(self, preprocessor, sample_transaction_data):
        """Test that inverse_transform works and returns DataFrame."""
        # Fit and transform
        features = preprocessor.fit_transform(sample_transaction_data)
        
        # Inverse transform
        recovered = preprocessor.inverse_transform(features)
        
        # Check that result is a DataFrame
        assert isinstance(recovered, pd.DataFrame)
        
        # Check shape matches original
        assert recovered.shape == sample_transaction_data.shape
        
        # Check columns match original
        assert list(recovered.columns) == list(sample_transaction_data.columns)
    
    def test_inverse_transform_raises_error_when_not_fitted(self, preprocessor):
        """Test that inverse_transform raises error when called before fit."""
        features = np.random.rand(5, 5) * np.pi
        
        with pytest.raises(ValueError, match="Preprocessor must be fitted before inverse_transform"):
            preprocessor.inverse_transform(features)
    
    def test_inverse_transform_categorical_values(self, preprocessor, sample_transaction_data):
        """Test that categorical values are correctly recovered."""
        features = preprocessor.fit_transform(sample_transaction_data)
        recovered = preprocessor.inverse_transform(features)
        
        # Categorical columns should match original values
        for col in preprocessor.categorical_columns:
            assert list(recovered[col]) == list(sample_transaction_data[col])
    
    def test_inverse_transform_numerical_values(self, preprocessor, sample_transaction_data):
        """Test that numerical values are approximately recovered."""
        features = preprocessor.fit_transform(sample_transaction_data)
        recovered = preprocessor.inverse_transform(features)
        
        # Numerical columns should be close to original (within tolerance)
        for col in preprocessor.numerical_columns:
            np.testing.assert_array_almost_equal(
                recovered[col].values,
                sample_transaction_data[col].values,
                decimal=10
            )
    
    def test_round_trip_property(self, preprocessor, sample_transaction_data):
        """Test round-trip property: transform → inverse → transform produces equivalent results."""
        # First transformation
        features1 = preprocessor.fit_transform(sample_transaction_data)
        
        # Inverse transform
        recovered = preprocessor.inverse_transform(features1)
        
        # Second transformation
        features2 = preprocessor.transform(recovered)
        
        # Results should be very close (within numerical tolerance)
        np.testing.assert_array_almost_equal(features1, features2, decimal=10)
    
    def test_inverse_transform_with_only_categorical(self):
        """Test inverse_transform with only categorical columns."""
        data = pd.DataFrame({
            'category1': ['A', 'B', 'C', 'A', 'B'],
            'category2': ['X', 'Y', 'X', 'Z', 'Y']
        })
        
        preprocessor = TransactionPreprocessor(['category1', 'category2'], [])
        features = preprocessor.fit_transform(data)
        recovered = preprocessor.inverse_transform(features)
        
        assert list(recovered['category1']) == list(data['category1'])
        assert list(recovered['category2']) == list(data['category2'])
    
    def test_inverse_transform_with_only_numerical(self):
        """Test inverse_transform with only numerical columns."""
        data = pd.DataFrame({
            'value1': [10.0, 20.0, 30.0, 40.0, 50.0],
            'value2': [1.5, 2.5, 3.5, 4.5, 5.5],
            'value3': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        
        preprocessor = TransactionPreprocessor([], ['value1', 'value2', 'value3'])
        features = preprocessor.fit_transform(data)
        recovered = preprocessor.inverse_transform(features)
        
        for col in ['value1', 'value2', 'value3']:
            np.testing.assert_array_almost_equal(
                recovered[col].values,
                data[col].values,
                decimal=10
            )
    
    def test_inverse_transform_preserves_column_order(self, preprocessor, sample_transaction_data):
        """Test that inverse_transform preserves original column order."""
        features = preprocessor.fit_transform(sample_transaction_data)
        recovered = preprocessor.inverse_transform(features)
        
        # Column order should match
        expected_columns = preprocessor.categorical_columns + preprocessor.numerical_columns
        assert list(recovered.columns) == expected_columns


class TestSaveLoad:
    """Tests for save and load methods."""
    
    def test_save_creates_file(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that save creates a file at the specified path."""
        # Fit the preprocessor
        preprocessor.fit(sample_transaction_data)
        
        # Save to temporary file
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Check that file was created
        assert save_path.exists()
        assert save_path.is_file()
    
    def test_save_raises_error_when_not_fitted(self, preprocessor, tmp_path):
        """Test that save raises error when called before fit."""
        save_path = tmp_path / "preprocessor.pkl"
        
        with pytest.raises(ValueError, match="Cannot save unfitted preprocessor"):
            preprocessor.save(str(save_path))
    
    def test_save_creates_parent_directories(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that save creates parent directories if they don't exist."""
        # Fit the preprocessor
        preprocessor.fit(sample_transaction_data)
        
        # Save to nested path that doesn't exist
        save_path = tmp_path / "models" / "preprocessor" / "v1" / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Check that file and directories were created
        assert save_path.exists()
        assert save_path.is_file()
    
    def test_load_restores_preprocessor(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that load successfully restores a preprocessor."""
        # Fit and save
        preprocessor.fit(sample_transaction_data)
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        loaded_preprocessor = TransactionPreprocessor.load(str(save_path))
        
        # Check that loaded preprocessor is fitted
        assert loaded_preprocessor.is_fitted is True
    
    def test_load_raises_error_for_nonexistent_file(self):
        """Test that load raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError, match="Preprocessor state file not found"):
            TransactionPreprocessor.load("/nonexistent/path/preprocessor.pkl")
    
    def test_load_restores_column_specifications(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that load restores categorical and numerical column specifications."""
        # Fit and save
        preprocessor.fit(sample_transaction_data)
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        loaded_preprocessor = TransactionPreprocessor.load(str(save_path))
        
        # Check column specifications
        assert loaded_preprocessor.categorical_columns == preprocessor.categorical_columns
        assert loaded_preprocessor.numerical_columns == preprocessor.numerical_columns
    
    def test_load_restores_scaler_state(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that load restores the fitted MinMaxScaler."""
        # Fit and save
        features_original = preprocessor.fit_transform(sample_transaction_data)
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        loaded_preprocessor = TransactionPreprocessor.load(str(save_path))
        
        # Transform with loaded preprocessor
        features_loaded = loaded_preprocessor.transform(sample_transaction_data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(features_original, features_loaded)
    
    def test_load_restores_label_encoders(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that load restores all fitted LabelEncoders."""
        # Fit and save
        preprocessor.fit(sample_transaction_data)
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        loaded_preprocessor = TransactionPreprocessor.load(str(save_path))
        
        # Check that label encoders are restored
        assert len(loaded_preprocessor.label_encoders) == len(preprocessor.categorical_columns)
        
        for col in preprocessor.categorical_columns:
            assert col in loaded_preprocessor.label_encoders
            # Check that encoder classes match
            np.testing.assert_array_equal(
                loaded_preprocessor.label_encoders[col].classes_,
                preprocessor.label_encoders[col].classes_
            )
    
    def test_load_restores_imputers(self, preprocessor, tmp_path):
        """Test that load restores numerical and categorical imputers."""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'merchant_category': ['online_retail', None, 'gas_station', 'online_retail', 'grocery'],
            'card_type': ['visa', 'mastercard', None, 'amex', 'visa'],
            'amount': [100.0, np.nan, 30.0, 200.0, 75.0],
            'time_of_day': [14.5, 9.0, np.nan, 22.0, 12.0],
            'location_distance': [5.0, 2.0, 10.0, np.nan, 3.0]
        })
        
        # Fit and save
        features_original = preprocessor.fit_transform(data_with_missing)
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        loaded_preprocessor = TransactionPreprocessor.load(str(save_path))
        
        # Transform with loaded preprocessor
        features_loaded = loaded_preprocessor.transform(data_with_missing)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(features_original, features_loaded)
    
    def test_round_trip_property_for_persistence(self, preprocessor, sample_transaction_data, tmp_path):
        """Test round-trip property: save then load produces equivalent predictions (Requirement 6.5)."""
        # Fit original preprocessor
        features_original = preprocessor.fit_transform(sample_transaction_data)
        
        # Save
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        loaded_preprocessor = TransactionPreprocessor.load(str(save_path))
        
        # Transform with loaded preprocessor
        features_loaded = loaded_preprocessor.transform(sample_transaction_data)
        
        # Results should be equivalent within numerical tolerance
        np.testing.assert_array_almost_equal(features_original, features_loaded, decimal=10)
    
    def test_load_works_with_new_data(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that loaded preprocessor can transform new unseen data."""
        # Fit and save with original data
        preprocessor.fit(sample_transaction_data)
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Create new data with same schema and values within training range
        # Training ranges: amount [30, 200], time_of_day [9, 22], location_distance [2, 50]
        new_data = pd.DataFrame({
            'merchant_category': ['grocery', 'online_retail', 'gas_station'],
            'card_type': ['visa', 'mastercard', 'visa'],
            'amount': [60.0, 150.0, 40.0],
            'time_of_day': [10.0, 16.5, 20.0],
            'location_distance': [4.0, 25.0, 8.0]
        })
        
        # Load and transform new data
        loaded_preprocessor = TransactionPreprocessor.load(str(save_path))
        features = loaded_preprocessor.transform(new_data)
        
        # Check that transformation works correctly
        assert features.shape == (3, 5)
        assert np.all(features >= 0)
        assert np.all(features <= np.pi)
    
    def test_load_handles_new_categorical_values(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that loaded preprocessor handles new categorical values gracefully."""
        # Fit and save with original data
        preprocessor.fit(sample_transaction_data)
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Create new data with unseen categorical value
        new_data = pd.DataFrame({
            'merchant_category': ['restaurant'],  # New category not in training
            'card_type': ['visa'],
            'amount': [60.0],
            'time_of_day': [10.0],
            'location_distance': [1.0]
        })
        
        # Load preprocessor
        loaded_preprocessor = TransactionPreprocessor.load(str(save_path))
        
        # Transform should raise an error for unseen categorical value
        with pytest.raises(ValueError):
            loaded_preprocessor.transform(new_data)
    
    def test_multiple_save_load_cycles(self, preprocessor, sample_transaction_data, tmp_path):
        """Test that multiple save/load cycles maintain consistency."""
        # Fit original
        preprocessor.fit(sample_transaction_data)
        
        # Multiple save/load cycles
        for i in range(3):
            save_path = tmp_path / f"preprocessor_{i}.pkl"
            preprocessor.save(str(save_path))
            preprocessor = TransactionPreprocessor.load(str(save_path))
        
        # Final transformation should still work correctly
        features = preprocessor.transform(sample_transaction_data)
        assert features.shape == (5, 5)
        assert np.all(features >= 0)
        assert np.all(features <= np.pi)
    
    def test_save_with_only_categorical_columns(self, sample_transaction_data, tmp_path):
        """Test save/load with only categorical columns."""
        categorical_only_data = sample_transaction_data[['merchant_category', 'card_type']]
        preprocessor = TransactionPreprocessor(['merchant_category', 'card_type'], [])
        
        # Fit and save
        preprocessor.fit(categorical_only_data)
        save_path = tmp_path / "preprocessor_categorical.pkl"
        preprocessor.save(str(save_path))
        
        # Load and transform
        loaded = TransactionPreprocessor.load(str(save_path))
        features = loaded.transform(categorical_only_data)
        
        assert features.shape == (5, 2)
    
    def test_save_with_only_numerical_columns(self, sample_transaction_data, tmp_path):
        """Test save/load with only numerical columns."""
        numerical_only_data = sample_transaction_data[['amount', 'time_of_day', 'location_distance']]
        preprocessor = TransactionPreprocessor([], ['amount', 'time_of_day', 'location_distance'])
        
        # Fit and save
        preprocessor.fit(numerical_only_data)
        save_path = tmp_path / "preprocessor_numerical.pkl"
        preprocessor.save(str(save_path))
        
        # Load and transform
        loaded = TransactionPreprocessor.load(str(save_path))
        features = loaded.transform(numerical_only_data)
        
        assert features.shape == (5, 3)
