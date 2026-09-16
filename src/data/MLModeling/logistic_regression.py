import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_preprocessed_data():
    train_path = r"D:\PythonProject\Placement_Prediction_System\data\preprocessed_train.csv"
    test_path = r"D:\PythonProject\Placement_Prediction_System\data\preprocessed_test.csv"
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data,test_data

def split_features_target(train_data,test_data):
    x_train = train_data.drop(columns=["PlacementStatus"])
    y_train = train_data["PlacementStatus"]
    x_test = test_data.drop(columns=["PlacementStatus"])
    y_test = test_data["PlacementStatus"]
    return x_train,x_test,y_train,y_test

def create_model():
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("\nAccuracy:")
    print(model.score(x_test, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model):
    model_path = r"D:\PythonProject\Placement_Prediction_System\models\logistic_regression_model.pkl"
    joblib.dump(model, model_path)
    print("\nModel saved successfully")
    print(model_path)

if __name__ == "__main__":

    train_data, test_data = load_preprocessed_data()

    print("Training Data Shape:")
    print(train_data.shape)

    print("\nTesting Data Shape:")
    print(test_data.shape)

    # Split features and target
    x_train, x_test, y_train, y_test = split_features_target(
        train_data,
        test_data
    )

    # Create model
    model = create_model()

    # Train model
    model = train_model(model, x_train, y_train)

    # Evaluate model
    evaluate_model(model, x_test, y_test)

    # Save model
    save_model(model)

