import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np


def getData():
    data = pd.read_csv("data/House_Prices.csv")
    return data


def preparation_Data():
    data = getData()
    
    data["MSSubClass"] = data["MSSubClass"].astype(str)

    numeric_cols = data.select_dtypes(
        include=["int64", "float64"]
    ).columns

    
    categorical_cols = data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    for col in numeric_cols:
        if col != "SalePrice":
            data[col] = data[col].fillna(data[col].median())

    for col in categorical_cols:
        data[col] = data[col].fillna("Unknown")

    data.to_csv("data/House_PricesV1.csv",  index=False)

    data = pd.get_dummies(
        data,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Préparation terminée.")
    print("X_train :", X_train.shape)
    print("X_test  :", X_test.shape)
    print("y_train :", y_train.shape)
    print("y_test  :", y_test.shape)

    df_rounds = data[data["SalePrice"] % 1 == 0].copy()
    df_rounds.to_csv("data/saleprice_ronds.csv", index=False)
    



def analysis():
    data = pd.read_csv("data/saleprice_ronds.csv")
    dataV1 = pd.read_csv("data/House_PricesV1.csv")
    
    
    
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # sns.histplot(data["SalePrice"], kde=True, ax=axes[0])
    # axes[0].set_title("SalePrice (raw)")

    # sns.histplot(np.log1p(data["SalePrice"]), kde=True, ax=axes[1])
    # axes[1].set_title("SalePrice (log1p)")

    # plt.tight_layout()
    # plt.show()

    # outliers = data[(data["GrLivArea"] < 4000) & (data["SalePrice"] < 200000)]
    # data_filtered = data[
    #     ~((data["GrLivArea"] > 4000) &
    #     (data["SalePrice"] < 300000))
    # ]

    # plt.figure(figsize=(10, 6))

    # sns.scatterplot(
    #     data=data_filtered,
    #     x="GrLivArea",
    #     y="SalePrice"
    # )

    # plt.title("Surface habitable vs prix")
    # plt.xlabel("Surface habitable")
    # plt.ylabel("Prix")

    # plt.show()

    # plt.figure(figsize=(10, 6))

    # sns.boxplot(
    #     data=data,
    #     x="OverallQual",
    #     y="SalePrice"
    # )

    # plt.title("Qualité du logement vs prix")
    # plt.xlabel("Qualité globale")
    # plt.ylabel("Prix")

    # plt.show()

    # plt.figure(figsize=(10, 6))

    # sns.scatterplot(
    #     data=data,
    #     x="YearBuilt",
    #     y="SalePrice"
    # )

    # plt.title("Année de construction vs prix")
    # plt.xlabel("Année de construction")
    # plt.ylabel("Prix")

    # plt.show()

    numeric_df = dataV1.select_dtypes(include="number").drop(columns=["Id"], errors="ignore")
    corr = numeric_df.corr()
    plt.figure(figsize=(14, 10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )

    plt.title("Matrice de corrélation")

    plt.show()


    

analysis()