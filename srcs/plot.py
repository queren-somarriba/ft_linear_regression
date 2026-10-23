import numpy as np
import matplotlib.pyplot as plt
from predict import get_thetas
from train import load_data


def main() -> None:
    """
    Plot the dataset (km vs price) and the regression line
    computed from thetas.json
    """
    data = load_data("data.csv")
    X = data['km'].to_numpy(dtype=float)
    y = data['price'].to_numpy(dtype=float)
    theta0, theta1 = get_thetas()

    xs = np.linspace(X.min(), X.max(), 100)
    ys = theta0 + theta1 * xs

    plt.scatter(X, y, label="Data")
    plt.plot(xs, ys, color="red",
             label=f"y = {theta0:.2f} + {theta1:.5f} * x")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car price vs mileage")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
