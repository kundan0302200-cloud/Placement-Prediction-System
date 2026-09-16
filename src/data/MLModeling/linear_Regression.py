from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)


def create_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5)
    }

    return models


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def predict(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def main():

    # --------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------
    df = load_data()

    print("\nOriginal Dataset Shape")
    print(df.shape)

    # --------------------------------------------------
    # 2. Train-Test Split
    # --------------------------------------------------
    x_train, x_test, y_train, y_test = split_data(
        df,
        target_column="Salary Package",
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "IsAnomaly"
        ]
    )

    print("\nTraining Shape")
    print(x_train.shape)

    print("\nTesting Shape")
    print(x_test.shape)

    # --------------------------------------------------
    # 3. Identify Numerical and Categorical Features
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

    print("One-Hot Encoding Completed.")

    # --------------------------------------------------
    # 8. Ordinal Encoding
    # --------------------------------------------------
    x_train, x_test, ordinal_encoder = ordinal_encode_data(
        x_train,
        x_test,
        ordinal_features
    )

    print("Ordinal Encoding Completed.")

    # --------------------------------------------------
    # 9. Create Models
    # --------------------------------------------------
    models = create_models()

    print("\nModels Created:")
    for model_name in models:
        print("-", model_name)

    # --------------------------------------------------
    # 10. Train, Predict and Evaluate Models
    # --------------------------------------------------
    results = {}

    for model_name, model in models.items():

        print(f"\n{'=' * 50}")
        print(f"Training {model_name}")
        print(f"{'=' * 50}")

        # Train
        trained_model = train_model(
            model,
            x_train,
            y_train
        )

        # Predict
        y_pred = predict(
            trained_model,
            x_test
        )

        # Evaluate
        metrics = evaluate_model(
            y_test,
            y_pred
        )

        results[model_name] = metrics

        print(f"MAE  : {metrics['MAE']:.4f}")
        print(f"MSE  : {metrics['MSE']:.4f}")
        print(f"RMSE : {metrics['RMSE']:.4f}")
        print(f"R²   : {metrics['R2']:.4f}")

    # --------------------------------------------------
    # 11. Display Final Comparison
    # --------------------------------------------------
    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<20}"
        f"{'MAE':<12}"
        f"{'MSE':<15}"
        f"{'RMSE':<12}"
        f"{'R²':<10}"
    )

    print("-" * 70)

    for model_name, metrics in results.items():

        print(
            f"{model_name:<20}"
            f"{metrics['MAE']:<12.4f}"
            f"{metrics['MSE']:<15.4f}"
            f"{metrics['RMSE']:<12.4f}"
            f"{metrics['R2']:<10.4f}"
        )

    # --------------------------------------------------
    # 12. Find Best Model
    # --------------------------------------------------
    best_model_name = max(
        results,
        key=lambda model: results[model]["R2"]
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(f"Best Model: {best_model_name}")
    print(f"R² Score : {results[best_model_name]['R2']:.4f}")
    print(f"RMSE     : {results[best_model_name]['RMSE']:.4f}")


if __name__ == "__main__":
    main()