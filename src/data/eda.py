import pandas as pd
import seaborn as sns
from src.data.load_data import load_data
import matplotlib.pyplot as plt
def basic_eda(df):

    print("First five rows of the dataset")
    print(df.head())
    print("Last five rows of the dataset")
    print(df.tail())
    print("25 to 35 rows of the dataset")
    print(df.iloc[25:36])
    print("Sample 10 records of the dataset")
    print(df.sample(10))
    print("Column names of the dataset")
    print(df.columns)
    print("=" * 50)
    print("datatypes of all the features")
    print(df.dtypes)
    print("=" * 50)
    print("Complete information of the dataset")
    print(df.info())
    print("=" * 50)
    print("description of the dataset")
    print(df.describe())
    print("=" * 50)
    print("number of null values in the dataset")
    print(df.isnull().sum().count())
    print("=" * 50)
    print("Number of duplicate values in the dataset")
    print(df.duplicated().sum())
    print("=" * 50)
    print(df["PlacementStatus"].value_counts())
    print("=" * 50)
    count = df["PlacementStatus"].value_counts()
    plt.figure(figsize = (6,5))
    plt.bar(count.index,count.values)
    print("=" * 50)
    plt.title("distribution of placement status")
    plt.xlabel("placement status")
    plt.ylabel("number of placement status")
    plt.savefig(r"D:\PythonProject\Placement_Prediction_System\results\placement_status_bar.png")
    plt.show()

def univariate(df):
    plt.figure(figsize = (6,5))
    plt.hist(df["CGPA"], bins = 10, edgecolor = "black")
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("frequency")
    plt.savefig(r"D:\PythonProject\Placement_Prediction_System\app\static\charts\CGPA_Histogram.png")
    plt.show()
    gendercount = df["Gender"].value_counts()
    plt.figure(figsize = (6,5))
    plt.pie(gendercount, labels = gendercount.index, autopct = "%1.1f%%", startangle = 90)
    plt.title("Gender distribution pie-chart")
    plt.savefig(r"D:\PythonProject\Placement_Prediction_System\app\static\charts\Gender_distribution_pie_chart.png")
    plt.show()

def bivariate(df):
    plt.figure(figsize = (6,5))
    plt.scatter(df["CGPA"], df["AptitudeTestScore"], color = "red")
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")
    plt.savefig(r"D:\PythonProject\Placement_Prediction_System\app\static\charts\CGPA_vs_Aptitude_test_score.png")
    plt.show()
    plt.close()

    plt.figure(figsize = (6,5))
    placed=df[df["PlacementStatus"] == 1]["CGPA"]
    not_placed=df[df["PlacementStatus"] == 0]["CGPA"]
    plt.boxplot([placed, not_placed], label = ["placed", "not placed"])
    plt.title("CGPA vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("CGPA")
    plt.savefig(r"D:\PythonProject\Placement_Prediction_System\app\static\charts\CGPA_vs_Placement_status.png")
    plt.show()
    plt.close()

def multivariate(df):
    data = df[["CGPA", "AptitudeTestScore", "PlacementStatus"]]
    correlation = data.corr()
    plt.figure(figsize = (6,5))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(r"D:\PythonProject\Placement_Prediction_System\app\static\charts\Correlation_Heatmap.png")
    plt.show()
    plt.close()

    correlation = df.corr(numeric_only=True)
    plt.figure(figsize = (8,6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(r"D:\PythonProject\Placement_Prediction_System\app\static\charts\Correlation_Heatmap1.png")
    plt.show()
    plt.close()




if __name__ == "__main__":
    df = load_data()
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)