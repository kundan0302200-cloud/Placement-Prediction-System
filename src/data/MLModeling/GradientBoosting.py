import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ============================================================
# 1. LOAD DATA
# ============================================================

df = load_data()

print("Original Dataset Shape:")
print(df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

target_column = "Salary Package"


# ============================================================
# 3. DROP UNNECESSARY COLUMNS
# ============================================================

drop_columns = [
    "StudentID",
    "PlacementStatus",
    "IsAnomaly"
]

drop_columns = [
    col for col in drop_columns
    if col in df.columns
]

X = df.drop(columns=drop_columns + [target_column])
y = df[target_column]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)


# ============================================================
# 5. IDENTIFY FEATURES
# ============================================================

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train = X_train.copy()
X_test = X_test.copy()

X_train[numerical_features] = imputer.fit_transform(
    X_train[numerical_features]
)

X_test[numerical_features] = imputer.transform(
    X_test[numerical_features]
)


# ============================================================
# 7. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(
    X_train[numerical_features]
)

X_test[numerical_features] = scaler.transform(
    X_test[numerical_features]
)


# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

one_hot_features = [
    "Gender",
    "City",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs"
]

one_hot_features = [
    col for col in one_hot_features
    if col in X_train.columns
]

if len(one_hot_features) > 0:

    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_encoded = one_hot_encoder.fit_transform(
        X_train[one_hot_features]
    )

    test_encoded = one_hot_encoder.transform(
        X_test[one_hot_features]
    )

    encoded_columns = one_hot_encoder.get_feature_names_out(
        one_hot_features
    )

    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )

    X_train = X_train.drop(
        columns=one_hot_features
    )

    X_test = X_test.drop(
        columns=one_hot_features
    )

    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )


# ============================================================
# 9. ORDINAL ENCODING
# ============================================================

ordinal_features = [
    "CollegeTier",
    "CGPA_Tier"
]

ordinal_features = [
    col for col in ordinal_features
    if col in X_train.columns
]

if len(ordinal_features) > 0:

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    train_encoded = ordinal_encoder.fit_transform(
        X_train[ordinal_features]
    )

    test_encoded = ordinal_encoder.transform(
        X_test[ordinal_features]
    )

    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=ordinal_features,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=ordinal_features,
        index=X_test.index
    )

    X_train = X_train.drop(
        columns=ordinal_features
    )

    X_test = X_test.drop(
        columns=ordinal_features
    )

    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )


# ============================================================
# 10. CONVERT TO NUMPY
# ============================================================

X_train = X_train.astype(float).values
X_test = X_test.astype(float).values

y_train = y_train.astype(float).values
y_test = y_test.astype(float).values


print("\nFinal Feature Shape:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# 11. ADD BIAS COLUMN
# ============================================================

# Adds x0 = 1 for intercept/bias
X_train = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]

X_test = np.c_[
    np.ones(X_test.shape[0]),
    X_test
]


# ============================================================
# 12. GRADIENT DESCENT
# ============================================================

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):

    m = X.shape[0]

    # Initialize weights with zeros
    theta = np.zeros(X.shape[1])

    cost_history = []

    for epoch in range(epochs):

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = X @ theta

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        error = predictions - y

        # ----------------------------------------------------
        # Cost Function
        # ----------------------------------------------------

        cost = (1 / (2 * m)) * np.sum(error ** 2)

        cost_history.append(cost)

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        gradient = (1 / m) * (X.T @ error)

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        theta = theta - learning_rate * gradient

        # Display progress
        if (epoch + 1) % 100 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Cost = {cost:.6f}"
            )

    return theta, cost_history


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n========================================")
print("GRADIENT DESCENT TRAINING")
print("========================================")

learning_rate = 0.01
epochs = 1000

theta, cost_history = gradient_descent(
    X_train,
    y_train,
    learning_rate=learning_rate,
    epochs=epochs
)


# ============================================================
# 14. PREDICTION
# ============================================================

y_pred = X_test @ theta


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========================================")
print("GRADIENT DESCENT RESULTS")
print("========================================")

print(f"MAE  : {mae:.5f}")
print(f"MSE  : {mse:.5f}")
print(f"RMSE : {rmse:.5f}")
print(f"R²   : {r2:.5f}")


# ============================================================
# 16. SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Salary": y_test,
    "Predicted Salary": y_pred
})

print("\nSample Predictions:")
print(results.head(10))


# ============================================================
# 17. PLOT COST FUNCTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    cost_history
)

plt.xlabel("Epoch")
plt.ylabel("Cost (MSE / 2)")
plt.title("Gradient Descent Cost Reduction")

plt.grid(True)
plt.show()


# ============================================================
# 18. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual Salary Package")
plt.ylabel("Predicted Salary Package")

plt.title(
    "Actual vs Predicted Salary - Gradient Descent"
)

plt.grid(True)
plt.show()