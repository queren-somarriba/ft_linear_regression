import numpy as np
import pandas as pd
import os
from linearRegression import LinearRegression
# import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Read datas in file_path
    """
    if not os.path.exists(file_path):
        raise ValueError("Error: no file found\n")
    try:
        price_mileage = pd.read_csv(file_path, on_bad_lines='skip')
        return price_mileage
    except:
            print("Error in file")


def train_model():
    data = load_data("data.csv")
    X = np.array(data['km'])
    y = np.array(data['price'])
    rate = 0.1  # test 0.05 -> 2
    # costs = []
    # rates = []
    model = LinearRegression(learning_rate=rate)
    # while rate < 2.0:
    model.fit(X, y, iterations=100)
    model.save_thetas("thetas.json")
    # costs.append(model.cost)
    # rates.append(rate)
    # rate += 0.01
    # predictions = model.predict(X)
    # print(predictions)
    # plt.scatter(rates, costs)
    # plt.show()


if __name__ == "__main__":
    train_model()
