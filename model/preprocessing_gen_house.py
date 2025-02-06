import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class GeneralHousePriceDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        #drop date, id and zipcode columns due to irrelevancy
        self.data.drop(columns=["id", "date", "zipcode"], inplace=True, errors="ignore")
        
        #limit price and sft_lot to 2mil
        self.data["price"] = self.data["price"].astype("float32", errors="ignore")
        self.data = self.data[self.data["price"] <= 2000000]
        self.data = self.data[self.data["sqft_lot"] <= 2000000]

        #log-transform the price and sqft_lot
        self.data.dropna(subset=["price", "sqft_lot"], inplace=True)
        self.data["price"] = np.log(self.data["price"])
        self.data["sqft_lot"] = np.log(self.data["sqft_lot"])

        #split the columns in numeric and categorical
        self.numeric_columns = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "view",      
            "grade",
            "sqft_above",
            "sqft_basement",
            "yr_built",
            "yr_renovated",
            "lat",
            "long",
            "sqft_living15",
            "sqft_lot15"
        ]

        self.categorical_columns = [
            "waterfront",
            "condition"
        ]


        #one-hot encode the categorical columns
        self.data = pd.get_dummies(self.data, columns=self.categorical_columns)

        self.targets = self.data["price"].values.astype("float32")

        #scale the log(price) with StandardScaler
        self.scaler = StandardScaler()
        prices = self.data["price"].values.astype("float32").reshape(-1, 1)
        normalized_prices = self.scaler.fit_transform(prices)
        self.data["price"] = normalized_prices.flatten()

        self.targets = self.data["price"].values.astype("float32")

        feature_columns = self.data.columns.drop("price")
        self.features = self.data[feature_columns].values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.features[idx]
        y = self.targets[idx]

        if self.transform:
            X = self.transform(X)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
