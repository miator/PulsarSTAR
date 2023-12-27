import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Rename columns
    data = _rename_columns(data)

    # Split into features x and target y
    x, y = _split_features_target(data)

    # Train-test split
    x_train, x_test, y_train, y_test = _train_test_split(x, y)

    return x_train, x_test, y_train, y_test


def _rename_columns(data):
    columns = ['Mean P', 'StD P', 'Kurtosis P', 'Skewness P',
               'Mean DM_SNR', 'StD DM_SNR', 'Kurtosis DM_SNR', 'Skewness DM_SNR', 'Target']
    data.rename(columns=dict(zip(data.columns, columns)), inplace=True)
    return data


def _split_features_target(data):
    x = data.drop('Target', axis=1)
    y = data['Target']
    return x, y


def _train_test_split(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)
