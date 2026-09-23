from src.data.load_data import load_data
from src.data.preprocess import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import matplotlib.pyplot as plt

def create_model(k):
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    return model
def train_model(model, X):
    model.fit(X)
    print("\nAglomerative Clustering \n")
    return model

def evaluate_model(model, X):
    labels = model.labels_
    score = silhouette_score(X, labels)
    print("\nSilhouette score \n")
    print(score)
    return labels

def display_dendogram(X, cut_distance):
    linked = linkage(X, method="ward")

    plt.figure(figsize=(12, 6))

    dendrogram(
        linked,
        truncate_mode="lastp",
        p=30
    )

    plt.axhline(
        y=cut_distance,
        linestyle="--",
        color="r"
    )

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster / Data Points")
    plt.ylabel("Ward Distance")

    return linked

def main():
    df = load_data()

    print("Dataset shape")
    print(df.shape)

    # Take sample for clustering
    df = df.sample(n=500, random_state=42)

    X = split_X_data(
        df,
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    numerical_features, categorical_features = identify_features(X)

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

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

    # -------------------------------
    # 1. Handle Missing Values
    # -------------------------------
    X, _, imputer = handle_missing_values(
        X,
        X.copy(),
        numerical_features
    )

    # -------------------------------
    # 2. Standardize Numerical Data
    # -------------------------------
    X, _, scaler = standardize_data(
        X,
        X.copy(),
        numerical_features
    )

    # -------------------------------
    # 3. One-Hot Encoding
    # -------------------------------
    X, _, one_hot_encoder = one_hot_encode_data(
        X,
        X.copy(),
        one_hot_features
    )

    # -------------------------------
    # 4. Ordinal Encoding
    # -------------------------------
    X, _, ordinal_encoder = ordinal_encode_data(
        X,
        X.copy(),
        ordinal_features
    )

    # -------------------------------
    # Convert to numerical matrix
    # -------------------------------
    X = X.select_dtypes(include=["number"])

    print("\nFinal processed data shape:")
    print(X.shape)

    # -------------------------------
    # 5. Create Agglomerative Model
    # -------------------------------
    k = 3

    model = create_model(k)

    # -------------------------------
    # 6. Train Model
    # -------------------------------
    model = train_model(model, X)

    # -------------------------------
    # 7. Evaluate Model
    # -------------------------------
    labels = evaluate_model(model, X)

    # -------------------------------
    # 8. Display cluster information
    # -------------------------------
    print("\nCluster Labels:")
    print(labels)

    print("\nNumber of samples in each cluster:")

    unique, counts = __import__("numpy").unique(
        labels,
        return_counts=True
    )

    for cluster, count in zip(unique, counts):
        print("Cluster", cluster, ":", count)

    # -------------------------------
    # 9. Display Dendrogram
    # -------------------------------
    print("\nDisplaying Dendrogram...")

    display_dendogram(
        X,
        cut_distance=10
    )

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Ward Distance")
    plt.show()


if __name__ == "__main__":
    main()








