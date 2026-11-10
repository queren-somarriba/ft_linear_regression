import json
import os
import numpy as np


def get_thetas() -> tuple[float, float]:
    """Read and return theta0, theta1 from thetas.json"""
    if not os.path.exists("thetas.json"):
        return 0.0, 0.0
    try:
        with open("thetas.json", "r") as f:
            thetas = json.load(f)
            if not thetas:
                return 0.0, 0.0

            theta0 = float(thetas["theta0"])
            theta1 = float(thetas["theta1"])
        return theta0, theta1
    except json.JSONDecodeError:
        return 0.0, 0.0
    except (KeyError, TypeError, ValueError) as e:
        print(f"Warning: invalid theta values in thetas.json ({e}), \
              using theta0=0, theta1=0")
        return 0.0, 0.0


def get_mileage() -> float:
    """Ask user for a mileage, validate it, retry on invalid input."""
    while True:
        try:
            mileage = float(input("What the mileage of your car?\n>"))
            if mileage < 0:
                print("Error: mileage cannot be negative.")
                continue
            if np.isnan(mileage) or np.isinf(mileage):
                print("Error: mileage must be a finite value")
                continue
            return mileage
        except ValueError:
            print("Error: please enter a valid number.")


def predict_price(mileage: float) -> float:
    """
    Predict Mileage according to theta0 and theta1
    """
    theta0, theta1 = get_thetas()
    return theta0 + theta1 * mileage


def main() -> None:
    """
    Ask the mileage of the user car and predict the price
    with values from thetas.json
    """
    mileage = get_mileage()
    price = predict_price(mileage)
    print(f"The estimate price of your car is: {price:.2f}")


if __name__ == "__main__":
    main()
