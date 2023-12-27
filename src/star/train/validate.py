from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.star.data.dataset import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def hyperparameter_tuning(_filepath):
    # Load and preprocess the data
    x_train, x_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Split the training set into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the GridSearchCV to the training data
    grid_search.fit(x_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Use the best model for prediction on the validation set
    best_rf_model = grid_search.best_estimator_
    y_val_pred = best_rf_model.predict(x_val)

    # Evaluate the model on the validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    return best_rf_model


if __name__ == "__main__":
    filepath = r'C:\Users\User\Desktop\STAR\HTRU_2.csv'

    best_model = hyperparameter_tuning(filepath)
