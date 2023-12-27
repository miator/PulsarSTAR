from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.star.data.dataset import load_and_preprocess_data


def train_model(_x_train, _x_test, _y_train, _y_test):
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(_x_train, _y_train)  # Use the correct variable names

    # Make predictions on the test set
    y_pred = rf_model.predict(_x_test)

    # Evaluate the model
    accuracy = accuracy_score(_y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(_y_test, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)

    class_report = classification_report(_y_test, y_pred)
    print('Classification Report:')
    print(class_report)

    return rf_model


if __name__ == "__main__":
    filepath = r'C:\Users\User\Desktop\STAR\HTRU_2.csv'

    # Load and preprocess the data
    x_train, x_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Train the model
    trained_model = train_model(x_train, x_test, y_train, y_test)
