from src.data.load_data import load_data

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data.preprocess import (
    split_X_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# FIND OPTIMAL K USING ELBOW METHOD
# ---------------------------------------------------------
def find_optimal_k(X):
    wcss = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )

        model.fit(X)
        wcss.append(model.inertia_)

    print("\nWCSS values:")

    for k, value in zip(range(1, 11), wcss):
        print("K =", k, "WCSS =", value)

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, 11),
        wcss,
        marker="o"
    )

    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal K")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

    return wcss


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------
def create_model(k):

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42
    )

    return model


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
def train_model(model, X):

    labels = model.fit_predict(X)

    print("\nK-Means trained successfully")

    return model, labels


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------
def evaluate_model(model, X, labels):

    print("\nInertia (WCSS):")
    print(model.inertia_)

    print("\nIterations to Converge:")
    print(model.n_iter_)

    silhouette = silhouette_score(X, labels)

    print("\nSilhouette Score:")
    print(silhouette)

    return silhouette


# ---------------------------------------------------------
# DISPLAY CLUSTERS
# ---------------------------------------------------------
def display_clusters(X, labels, model):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels,
        cmap="viridis",
        s=30
    )

    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.title("K-Means Clustering")

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    plt.legend()
    plt.grid(True)

    plt.show()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    # -----------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------
    df = load_data()

    print("Original Dataset Shape:", df.shape)


    # -----------------------------------------------------
    # 2. SELECT FEATURES
    # -----------------------------------------------------
    X = split_X_data(
        df,
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nX Shape:", X.shape)


    # -----------------------------------------------------
    # 3. IDENTIFY FEATURES
    # -----------------------------------------------------
    numerical_features, categorical_features = identify_features(X)

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)


    # -----------------------------------------------------
    # 4. DEFINE CATEGORICAL FEATURES
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # 5. HANDLE MISSING VALUES
    # -----------------------------------------------------
    X, _, imputer = handle_missing_values(
        X,
        X.copy(),
        numerical_features
    )

    print("\nMissing Values Handling Completed")


    # -----------------------------------------------------
    # 6. STANDARDIZE NUMERICAL FEATURES
    # -----------------------------------------------------
    X, _, scaler = standardize_data(
        X,
        X.copy(),
        numerical_features
    )

    print("\nStandardization Completed")


    # -----------------------------------------------------
    # 7. ONE-HOT ENCODING
    # -----------------------------------------------------
    X, _, one_hot_encoder = one_hot_encode_data(
        X,
        X.copy(),
        one_hot_features
    )

    print("\nOne-Hot Encoding Completed")


    # -----------------------------------------------------
    # 8. ORDINAL ENCODING
    # -----------------------------------------------------
    X, _, ordinal_encoder = ordinal_encode_data(
        X,
        X.copy(),
        ordinal_features
    )

    print("\nOrdinal Encoding Completed")


    # -----------------------------------------------------
    # 9. CHECK FINAL DATA
    # -----------------------------------------------------
    print("\nFinal X Shape:", X.shape)

    print("\nFinal Data Types:")
    print(X.dtypes)

    print("\nChecking Missing Values:")
    print(X.isnull().sum().sum())


    # -----------------------------------------------------
    # 10. FIND OPTIMAL K
    # -----------------------------------------------------
    find_optimal_k(X)


    # -----------------------------------------------------
    # 11. SELECT K
    # -----------------------------------------------------
    k = 3

    print("\nSelected K =", k)


    # -----------------------------------------------------
    # 12. CREATE MODEL
    # -----------------------------------------------------
    model = create_model(k)


    # -----------------------------------------------------
    # 13. TRAIN MODEL
    # -----------------------------------------------------
    model, labels = train_model(
        model,
        X
    )


    # -----------------------------------------------------
    # 14. EVALUATE MODEL
    # -----------------------------------------------------
    evaluate_model(
        model,
        X,
        labels
    )


    # -----------------------------------------------------
    # 15. DISPLAY CLUSTERS
    # -----------------------------------------------------
    display_clusters(
        X,
        labels,
        model
    )


    # -----------------------------------------------------
    # 16. ADD CLUSTER TO DATASET
    # -----------------------------------------------------
    df["Cluster"] = labels

    print("\nDataset with Cluster Labels:")
    print(df[["StudentID", "Cluster"]].head())


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    main()