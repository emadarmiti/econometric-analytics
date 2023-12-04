import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_price_data(data: pd.DataFrame) -> None:
    """Create two scatter plots, for price-demand and price-revenue relationships.

    The input data should has the following columns: price, demand, and revenue.
    Args:
        data: price data has prices, demands and revenues
    """
    # create the figure
    plt.figure(figsize=(16, 6))

    # subplot for price-demand
    plt.subplot(1, 2, 1)
    plt.scatter(data["price"], data["demand"])
    plt.title("Price and Demand Relationship")
    plt.xlabel("Price")
    plt.ylabel("Demand")

    # subplot for price-revenue
    plt.subplot(1, 2, 2)
    plt.scatter(data["price"], data["revenue"])
    plt.title("Price and Revenue Relationship")
    plt.xlabel("Price")
    plt.ylabel("Revenue")

    plt.show()


def plot_predictions(
    data: pd.DataFrame,
    pred_price_range: np.ndarray,
    pred_demand: list,
    pred_revenue: np.ndarray,
) -> None:
    """Create two plots to show the predicted demands and revenue with optimal price, it also plots the original data.

    The input data should has the following columns: price, demand, and revenue.
    Args:
        data: price data has prices, demands and revenues
        pred_price_range: array of generated prices correspond to predicted values
        pred_demand: predicted demands for the pred_price_range
        pred_revenue: derived revenues from predicted demands predicted demands for the pred_price_range
    """
    # create figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # plot ground truth prices and demands
    ax1.scatter(data["price"], data["demand"], label="Ground Truth", color="blue")
    # plot predicted demands
    ax1.plot(pred_price_range, pred_demand, label="Predictions", color="red")

    # add a vertical line on the optimal price, take index of max revenue
    ax1.axvline(
        x=pred_price_range[pred_revenue.argmax()],
        color="green",
        linestyle="--",
        label=f"Optimal Price={round(pred_price_range[pred_revenue.argmax()], 1)}\n"
        + f"With Demand={round(pred_demand[pred_revenue.argmax()])}",
    )
    ax1.set_title("Price and Demand Relationship")
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Demand")
    ax1.legend()

    # plot ground truth prices and revenue
    ax2.scatter(data["price"], data["revenue"], label="Ground Truth", color="blue")
    # plot predicted revenues
    ax2.plot(
        pred_price_range, pred_revenue, label="Derived From Predictions", color="red"
    )

    # add a vertical line on the optimal price, take index of max revenue
    ax2.axvline(
        x=pred_price_range[pred_revenue.argmax()],
        color="green",
        linestyle="--",
        label=f"Optimal Price={round(pred_price_range[pred_revenue.argmax()], 1)}\n"
        + f"With Revenue={round(pred_revenue[pred_revenue.argmax()])}",
    )
    ax2.set_title("Price and Revenue Relationship")
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Revenue")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def predict_piecewise(
    price: float, const: float, alpha1: float, alpha2: float, threshold: float
) -> float:
    """Predict the demand based on a piecewise regression model.

    Calculates the demand for a price using two linear equations:
    one for prices <= threshold, and one for prices > threshold.
    The threshold is the breakpoint in the piecewise model.

    Args:
        price: price to predict demand
        const: the y intercept of the linear regression model for the first part
        alpha1: the slope of the linear regression model for the first part
        alpha2: the slope of the linear regression model for the second part
        threshold: the price that separates the two parts of the model

    Returns:
        float: the predicted demand
    """
    # if the price <= the threshold, use the first equation
    if price <= threshold:
        return const + alpha1 * price
    else:
        # calculate the demand at the threshold using the first equation
        demand_at_threshold = const + alpha1 * threshold
        # for prices > threshold start from the demand at the threshold
        # shift the starting points
        return demand_at_threshold + alpha2 * (price - threshold)


def exponential_decay_model(input: float, a: float, b: float) -> float:
    """Calculate the output of an exponential decreasing.

    This function represents an exponential decay process where the output decreases
    exponentially as the input value increases.
    Args:
    input: The input value to apply the equation on it
    a: The starting point of the decreasing process
    b: The degree of decreasing

    Returns:
        float: The calculated output of the exponential decay model
    """
    return a * np.exp(-b * input)
