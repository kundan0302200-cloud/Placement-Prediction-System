from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_data import load_data


def main():

    # Load dataset
    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)

    # Select numerical features
    X = df.select_dtypes(
        include=["int64", "float64"]
    )

    # Remove columns that should not be used
    columns_to_drop = [
        "StudentID",
        "Salary Package",
        "PlacementStatus",
        "IsAnomaly"
    ]

    X = X.drop(
        columns=[
            col for col in columns_to_drop
            if col in X.columns
        ]
    )

    print("\nFeatures used for K-Means:")
    print(X.columns.tolist())

    # Create K-Means model
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    # Fit model
    clusters = kmeans.fit_predict(X)

    # Add cluster labels
    df["Cluster"] = clusters

    # --------------------------------
    # Cluster information
    # --------------------------------

    print("\nCluster Counts:")
    print(
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nCluster Centers:")
    print(kmeans.cluster_centers_)

    print("\nInertia:")
    print(kmeans.inertia_)

    # --------------------------------
    # Plot 1: Cluster Count
    # --------------------------------

    cluster_counts = (
        df["Cluster"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values
    )

    plt.xlabel("Cluster")
    plt.ylabel("Number of Students")
    plt.title("Number of Students in Each Cluster")

    plt.show()

    # --------------------------------
    # Plot 2: K-Means Clusters
    # --------------------------------

    # Select first two numerical features
    feature1 = X.columns[0]
    feature2 = X.columns[1]

    plt.figure(figsize=(8, 6))

    for cluster in sorted(df["Cluster"].unique()):

        cluster_data = df[
            df["Cluster"] == cluster
        ]

        plt.scatter(
            cluster_data[feature1],
            cluster_data[feature2],
            label=f"Cluster {cluster}"
        )

    # Plot cluster centers
    centers = kmeans.cluster_centers_

    plt.scatter(
        centers[:, X.columns.get_loc(feature1)],
        centers[:, X.columns.get_loc(feature2)],
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title("K-Means Clustering")
    plt.legend()

    plt.show()

    # --------------------------------
    # Save result
    # --------------------------------



if __name__ == "__main__":
    main()