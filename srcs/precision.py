import numpy as np
from predict import get_thetas
from train import load_data


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MSE, RMSE, MAE and R2 between real and predicted prices
    """
    errors = y_pred - y_true
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def main() -> None:
    data = load_data("data.csv")
    X = data['km'].to_numpy(dtype=float)
    y = data['price'].to_numpy(dtype=float)
    theta0, theta1 = get_thetas()

    predictions = theta0 + theta1 * X
    metrics = compute_metrics(y, predictions)

    print(f"theta0 = {theta0:.4f}, theta1 = {theta1:.6f}")
    print(f"MSE  : {metrics['MSE']:.2f}")
    print(f"RMSE : {metrics['RMSE']:.2f} (average error in price units)")
    print(f"MAE  : {metrics['MAE']:.2f}")
    print(f"R2   : {metrics['R2']:.4f} (1.0 = perfect fit)")


if __name__ == "__main__":
    main()
