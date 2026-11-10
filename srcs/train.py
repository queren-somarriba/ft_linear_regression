import numpy as np
import pandas as pd
import os
from linearRegression import LinearRegression


def load_data(file_path):
    """
    Read datas in file_path
    """
    if not os.path.exists(file_path):
        raise ValueError("Error: no file found\n")
    try:
        price_mileage = pd.read_csv(file_path, on_bad_lines='error')
        return price_mileage
    except ValueError as e:
        print(f"Error: {e}\n")


def train_model():
    """
    load the data from data.csv, train the model with all the data and save
    thetas values to thetas.json
    """
    data = load_data("data.csv")
    X = np.array(data['km'])
    y = np.array(data['price'])
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Invalid data: Nan")
    rate = 0.05
    model = LinearRegression(learning_rate=rate)
    model.fit(X, y, iterations=1000)
    model.denormalize_thetas()
    model.save_thetas("thetas.json")


if __name__ == "__main__":
    train_model()
