from sklearn.discriminant_analysis import StandardScaler
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
import numpy as np

class DutchHousePriceDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        #ignore columns with Prijs op aanvraag
        self.data = self.data[~self.data["Price"].str.contains("Prijs op aanvraag", na=False)]

        self.data["Price"] = (
            self.data["Price"]
            .apply(
                lambda x: re.search(r"\d+", x.replace(".", "").replace(",", "")).group()
                if isinstance(x, str) 
                   and re.search(r"\d+", x.replace(".", "").replace(",", ""))
                else None
            )
            .astype("float32", errors="ignore")
        )

        #limit price to 2mil
        self.data = self.data[self.data["Price"] <= 2000000]

        #log-transform the price
        self.data = self.data.dropna(subset=["Price"])
        self.data["Price"] = np.log(self.data["Price"])

        #split the columns in numeric and categorical.
        self.numeric_columns = [
            "Lot size (m2)",
            "Living space size (m2)",
            "Build year",
            "Rooms",
            "Toilet",
            "Floors",
            "Estimated neighbourhood price per m2"
        ]
        self.categorical_columns = [
            "City",
            "Build type",
            "House type",
            "Roof",
            "Energy label",
            "Position",
            "Garden"
        ]

        irrelevant_columns = ["Address", "Postal code"]
        self.data.drop(columns=irrelevant_columns, inplace=True, errors="ignore")

        for col in self.numeric_columns:
            self.data[col] = (
                self.data[col]
                .astype(str)
                .str.replace("m²", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.extract(r"(\d+)")
                .fillna(-1)
                .astype("float32")
            )

        #one-hot encode
        self.data = pd.get_dummies(self.data, columns=self.categorical_columns)

        self.targets = self.data["Price"].values.astype("float32")

        #scale the log(price)
        self.scaler = StandardScaler()
        prices = self.data["Price"].values.astype("float32").reshape(-1, 1)
        normalized_prices = self.scaler.fit_transform(prices)
        self.data["Price"] = normalized_prices.flatten()

        self.targets = self.data["Price"].values.astype("float32")

        feature_columns = self.data.columns.drop("Price")
        self.features = self.data[feature_columns].values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.features[idx]
        y = self.targets[idx]

        if self.transform:
            X = self.transform(X)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
