import pytest
from src.star.train.train import train_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def test_train_model():
    # Generate a synthetic dataset for testing
    x, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(x_train, x_test, y_train, y_test)  # Pass all four arguments

    assert model is not None


if __name__ == '__main__':
    pytest.main([__file__])
