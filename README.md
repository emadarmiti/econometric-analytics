# econometric-analytics

# Introduction
This repository has two Data Science projects, the first one is related to price optimization and the other one is a time series forecast.

# Project 1: Optimal Price
### Overview
The goal of the project is to estimate the optimal price to maximize the revenue for a fashion item using survey data. The data was obtained from a survey of a group of approximately 2800 people, where the group was presented with different price levels, and then got willingness to buy the product. The data has a demand for at each price level. The solution uses machine learning and statistical aproaches to model the relationship between price and demand.
### File Structure  

```
├── datasets/
│   └── prices.csv
├── utils/
│   └── optimal_price_utils.py/
└── optimal_price.ipynb
```

The main code exists in `optimal_price.ipynb` Jupyter notebook, the notebook reads the datasets `prices.csv` from datasets folder. Also it uses `optimal_price_utils.py` Python file where all functions needed exist there.


# Project 2: Time Series Forecast
### Overview
The goal of the project is to analyze sales data for 7 different clothing products in terms of trends and seasonality and investigate correlations between different product types. Lastly, forecast the sales of the Dress product for the next five weeks. The data used is a sales dataset for Blouses, Dress, Hoodie, Jackets, Shorts, Skirt, and T-shirts products weekly from 2018-09-27 to 2020-09-10.
### File Structure  
```
├── datasets/
│   └── sales.csv
├── utils/
│   └── time_series_forecast_utils.py/
└── time_series_forecast.ipynb
```
The main code exists in `time_series_forecast.ipynb` Jupyter notebook, the notebook reads the datasets `sales.csv` from datasets folder. Also it uses `time_series_forecast_utils.py` Python file where all functions needed exist there.

# How To Use
To run the Jupyter notebooks you need to install the libraries exist in `requirements.txt`. You can follow the following instructions to run them a virtual environment. Run the following commands in your terminal after cloning the repository.

1. Create virtual environment
```bash
python -m venv venv
```

2. Activate virtual environment 
```bash
source myenv/bin/activate
```
3. Install dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
4. Run Jupyter notebook
```bash
jupyter notebook
```