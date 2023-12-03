import pandas as pd
import numpy as np
from scipy.stats import norm
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed


import warnings
warnings.filterwarnings('ignore')


def KPSS_trend_stationarity_test(time_series: pd.Series, title_name: str = "") -> None:
    """Run KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test to test if series has a trend.

    The null hypothesis is that the time series is stationary in terms of trend (no trend), if p-value
    Args:
        time_series: the series to run the test on
        title_name: series name to print in the results
    """
    # run the test and get the p-value
    kpss_p_value = kpss(time_series)[1]

    # if p-value is less than the significant level we reject the null hypothesis 
    if kpss_p_value < 0.05:
        print(
            f"""The {title_name} time series is not stationary in terms of trend based on KPSS test, p-value = {kpss_p_value} """
        )
    else:
        print(
            f"""The {title_name} time series is stationary in terms of trend based on KPSS test, p-value = {kpss_p_value} """
        )


def plot_time_series(time_series: pd.Series, title_name: str = "") -> None:
    """Plot time series.

    Args:
        time_series: the time series to plot.
        title_name: name of the series to add to plot.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(time_series.index, time_series, marker=".", linestyle="-", color="b", linewidth=1)
    plt.title(f"Weekly {title_name} Salse")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.show()


def autocorrelation_functions(time_series: pd.Series) -> None:
    """Plot autocorrelation function and partial autocorrelation function.

    Args:
        time_series: the series to plot autocorrelation function for it
    """
    # get number of lags to use, it should be < half of the data size
    number_lags = math.floor(time_series.shape[0] / 2) - 1

    # create one row and two columns figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # plot PACF
    plot_pacf(time_series, lags=number_lags, ax=axes[0])
    axes[0].set_title("PACF")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    # plot ACF
    plot_acf(time_series, lags=number_lags, ax=axes[1])
    axes[1].set_title("ACF")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Autocorrelation")
    plt.tight_layout()
    plt.show()


def STL_decomposition(time_series: pd.Series, plot: bool = True) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Decompose a time series and plot each component (trend, seasonality, residual).

    Args:
        time_series: the series to decompose
        plot: boolean to plot the components of the decomposed series or not

    Returns:
        trend, seasonality, and residual of the decomposed series
    """
    # decompose the series
    decomposed_series = STL(time_series).fit()

    # get it's components
    trend = decomposed_series.trend
    seasonal = decomposed_series.seasonal
    residual = decomposed_series.resid

    # if true plot trend, seasonality, and residual
    if plot:
        plt.figure(figsize=(10, 8))

        plt.subplot(4, 1, 1)
        plt.plot(time_series)
        plt.title("Original Series", fontsize=16)

        plt.subplot(4, 1, 2)
        plt.plot(trend)
        plt.title("Trend", fontsize=16)

        plt.subplot(4, 1, 3)
        plt.plot(seasonal)
        plt.title("Seasonal", fontsize=16)

        plt.subplot(4, 1, 4)
        plt.plot(residual, marker=".", linestyle="")
        plt.title("Residual", fontsize=16)

        plt.tight_layout()

    return trend, seasonal, residual


def analyze_time_series(time_series: pd.Series, plot_components: bool = True, plot_acf: bool = True) -> None:
    """Get all time series tools: STL decomposition, ACF, PACF, and run KPSS trend test.

    Args:
        time_series: the time series to get analysis for it
        plot_components: if to plot trend, seasonality, residuals
        plot_acf: if to run ACF and PACF
    """
    # decompose the series
    _ = STL_decomposition(time_series, plot=plot_components)

    # if to plot autocorrelation functions
    if plot_acf:
        autocorrelation_functions(time_series)

    # run the KPSS test to test for trend seasonality
    KPSS_trend_stationarity_test(time_series)


def evaluate_sarima_model(
    time_series: pd.Series,
    forecast_number: int,
    non_seasonal_params: tuple,
    seasonal_params: tuple,
    title_name: str = "",
    plot: bool = True,
) -> tuple[tuple, tuple, float]:
    """Train a SARIMA model, plot forecasts against ground truth, get MAE evaluation, check if model residuals are stationary.

    Args:
        time_series: the series to train the model on
        forecast_number: number of forecast to decide the size of testing 
        non_seasonal_params: the parameters for the ARIMA model
        seasonal_params: the parameters for the seasonal ARIMA model
        title_name: series name to add to plots
        plot: if to plot results
    Returns:
        non seasonal params, seasonal params, and MAE
    """
    # split the data based on number of forecasts
    training_data = time_series[:-forecast_number]
    testing_data = time_series[-forecast_number:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # fit sarima model, disp=False for disable logs
        sarima_model = SARIMAX(
            training_data, order=non_seasonal_params, seasonal_order=seasonal_params
        ).fit(disp=False)

    # make forecasts based on forecast_number, then round the numbers
    forecasts_list = round(sarima_model.forecast(steps=forecast_number))

    # convert the list to a series with testing data index
    forecasts = pd.Series(forecasts_list, index=testing_data.index)

    # get model residuals
    residuals = testing_data - forecasts

    # get MAE to get the error with actual data unit
    forecast_mean_absolute_error = mean_absolute_error(testing_data, forecasts)

    if plot:
        # plot the forecast with ground truth
        plt.figure(figsize=(12, 5))
        plt.plot(time_series, color="orange")
        plt.plot(forecasts, color="blue")
        plt.title(f"Last {forecast_number} Weeks Forecast for {title_name} Sales")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.xticks(rotation=45)
        plt.show()

        # plot model residuals
        plt.figure(figsize=(12, 5))
        plt.plot(residuals)
        plt.title("Model Residuals Over Time")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.xticks(rotation=45)
        plt.show()

        # print MAE
        print(f"\n\nmean absolute error: {forecast_mean_absolute_error}\n")

        # check model residual if it's stationary
        KPSS_trend_stationarity_test(residuals, title_name='residuals')

    return (non_seasonal_params, seasonal_params, forecast_mean_absolute_error)


def forecast_sarima_model(
    time_series: pd.Series,
    forecast_number: int,
    non_seasonal_params: tuple,
    seasonal_params: tuple,
    title_name: str = "",
) -> pd.Series:
    """Train a SARIMA model.

    Args:
        time_series: the series to train the model on
        forecast_number: number of forecast
        non_seasonal_params: the parameters for the ARIMA model
        seasonal_params: the parameters for the seasonal ARIMA model
        title_name: series name to add to plots
    Returns:
        forecast values
    """
    # get the date to forecast from, it will be after last date in original series by 1 time freq
    forecast_from_date = pd.date_range(
        start=time_series.index.max(), periods=2, freq=time_series.index.inferred_freq
    )[-1]

    # create the dates to forecast
    forcast_for_dates = pd.date_range(
        forecast_from_date, periods=5, freq=time_series.index.inferred_freq
    )

    # fit sarima model, disp=False for disable logs
    sarima_model = SARIMAX(
        time_series, order=non_seasonal_params, seasonal_order=seasonal_params
    ).fit(disp=False)

    # make forecasts based on forecast_number, then round the numbers
    forecasts_list = round(sarima_model.forecast(steps=forecast_number))

    # convert the list to a series with forecast index
    forecasts = pd.Series(forecasts_list, index=forcast_for_dates)

    # plot original series with forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(pd.concat([time_series.iloc[-1:], forecasts]), color="blue", label="Forecast")
    plt.plot(time_series.index, time_series, color="orange", label="Original Sales")
    plt.title(f"Weekly {title_name} Sales Including 5 Weeks Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

    # plot the forecast alone
    plt.figure(figsize=(12, 6))
    plt.plot(forecasts.index, forecasts, marker="o", linestyle="-", color="b", linewidth=1)
    plt.title(f"Forecasted 5 Weeks for {title_name} Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.show()

    return forecasts


def get_best_sarima_params(
    time_series: pd.Series,
    forecast_number: int,
    non_seasoanl_param_list: list,
    seasoanl_param_list: list,
) -> dict[str, tuple]:
    """Do hyper parameter tuning for SARIMA model, by evaluating different set of params in parallel.

    Args:
        time_series: time series to train on
        forecast_number: number of forecast, will decide the testing size
        non_seasoanl_param_list: ARIMA params
        seasoanl_param_list: seasonal ARIMA params

    Returns:
        dict of best non seasonal params, best seasonal params, and MAE
    """
    # Use Joblib's Parallel and delayed to parallelize the evaluate SARIMA models
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_sarima_model)(
            time_series=time_series,
            forecast_number=forecast_number,
            non_seasonal_params=non_seasonal_params,
            seasonal_params=seasonal_params,
            plot=False,
        )
        for non_seasonal_params in non_seasoanl_param_list
        for seasonal_params in seasoanl_param_list
    )

    # get best model params by min MAE
    best_params = min(results, key=lambda x: x[2])

    return {
        "best_non_seasonal_params": best_params[0],
        "best_seasonal_params": best_params[1],
        "mean_absolute_error": best_params[2],
    }


def cross_correlation(time_series1: pd.Series, time_series2: pd.Series) -> np.array:
    """Compute the cross correlation between two data series.

    Args:
        time_series1: first time series
        time_series2: second time series

    Returns:
         array of cross-correlation values and lags
    """
    cross_corr_positive = ccf(time_series1, time_series2, adjusted=False)
    cross_corr_negative = ccf(time_series2, time_series1, adjusted=False)[1:]
    return np.concatenate((cross_corr_negative[::-1], cross_corr_positive))


def cross_correlation_analysis(data: pd.DataFrame) -> list:
    """Analyze and plot the cross correlation for all unique pairs of columns in the data.

    The function will decompose each series to make it stationary and get the cross correlations based on residuals,
    non-stationary time series can show a correlation even when there is no true relationship,
    simply because both series may share a common trend or seasonality.
    Args:
        data: the input dataframe

    Returns:
        List of tuples containing significant correlations > 0.3 (abs), each tuple containing
          (column1, column2, lag, correlation value)
    """
    # create a list of unique pairs of columns
    column_pairs = list(itertools.combinations(data.columns, 2))

    significant_correlations = []

    # create the figure plot
    num_rows = (len(column_pairs) + 1) // 2
    plt.figure(figsize=(15, num_rows * 5))

    for index, (column1, column2) in enumerate(column_pairs, start=1):

        # for each pair get residuals using stl decomposition
        resid1 = STL_decomposition(data[column1], plot=False)[2]
        resid2 = STL_decomposition(data[column2], plot=False)[2]

        # get cross correlations of the residuals
        cross_corr = cross_correlation(resid1, resid2)

        # get standard error, and margin of error
        n = len(data)
        lags = np.arange(-n + 1, n)
        se = 1 / np.sqrt(n)
        z_score = norm.ppf(1 - 0.01 / 2)
        margin_error = z_score * se

        # if max corr > 0.3 (abs) return it
        max_corr = max(cross_corr, key=abs)
        if abs(max_corr) > 0.3 and abs(max_corr) > margin_error:
            max_corr_index = np.argmax(np.abs(cross_corr))
            significant_correlations.append(
                (column1, column2, lags[max_corr_index], max_corr)
            )

        # plot each pair
        plt.subplot(num_rows, 2, index)
        plt.stem(
            lags,
            cross_corr,
            linefmt="b-",
            markerfmt="bo",
            basefmt="k-",
            label="Cross-correlation",
        )
        plt.axhline(
            y=margin_error, color="red", linestyle="--", label="99% Confidence Interval"
        )
        plt.axhline(y=-margin_error, color="red", linestyle="--")
        plt.xlabel("Lag")
        plt.ylabel("Cross-Correlation Coefficient")
        plt.title(f"Cross-Correlation between {column1} and {column2}")
        plt.legend()

    plt.tight_layout()
    plt.show()

    return significant_correlations
