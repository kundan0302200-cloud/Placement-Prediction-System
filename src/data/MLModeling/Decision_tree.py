from sklearn.tree import DecisionTreeClassifier
from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)

from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
from sklearn import tree


def create_model():
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=5,
        random_state=42
    )
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)

    print("\nDecision Tree Model Trained Successfully\n")

    return model


def evaluate_model(model, x_test, y_test):

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    return y_pred


def display_tree(model, feature_names):

    plt.figure(figsize=(25, 12))

    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Not Placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title("ID3 Decision Tree - Placement Prediction")

    plt.show()


def main():

    # --------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------

    df = load_data()


    # --------------------------------------------------
    # 2. Split Data
    # --------------------------------------------------

    x_train, x_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]
    )


    # --------------------------------------------------
    # 3. Identify Features
    # --------------------------------------------------

    numerical_features, categorical_features = identify_features(x_train)

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)


    # --------------------------------------------------
    # 4. Define Encoding Features
    # --------------------------------------------------

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]


    # --------------------------------------------------
    # 5. Handle Missing Values
    # --------------------------------------------------

    x_train, x_test, imputer = handle_missing_values(
        x_train,
        x_test,
        numerical_features
    )

    print("\nMissing Value Handling Completed.")


    # --------------------------------------------------
    # 6. Standardization
    # --------------------------------------------------

    x_train, x_test, scaler = standardize_data(
        x_train,
        x_test,
        numerical_features
    )

    print("\nStandardization Completed.")


    # --------------------------------------------------
    # 7. One-Hot Encoding
    # --------------------------------------------------

    x_train, x_test, one_hot_encoder = one_hot_encode_data(
        x_train,
        x_test,
        one_hot_features
    )

    print("\nOne-Hot Encoding Completed.")


    # --------------------------------------------------
    # 8. Ordinal Encoding
    # --------------------------------------------------

    x_train, x_test, ordinal_encoder = ordinal_encode_data(
        x_train,
        x_test,
        ordinal_features
    )

    print("\nOrdinal Encoding Completed.")


    # --------------------------------------------------
    # 9. Create Model
    # --------------------------------------------------

    model = create_model()


    # --------------------------------------------------
    # 10. Train Model
    # --------------------------------------------------

    model = train_model(
        model, x_train, y_train
    )


    # --------------------------------------------------
    # 11. Evaluate Model
    # --------------------------------------------------

    y_pred = evaluate_model(
        model,
        x_test,
        y_test
    )


    # --------------------------------------------------
    # 12. Display Decision Tree
    # --------------------------------------------------

    display_tree(
        model,
        x_train.columns
    )


if __name__ == "__main__":
    main()