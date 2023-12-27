import pandas as pd
import pytest
from src.star.data.dataset import load_and_preprocess_data

TEST_FILEPATH = r'C:\Users\User\Desktop\STAR\HTRU_2.csv'


def test_load_and_preprocess_data():
    # Load and preprocess the test data
    x_train, x_test, y_train, y_test = load_and_preprocess_data(TEST_FILEPATH)

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


if __name__ == '__main__':
    pytest.main([__file__])
