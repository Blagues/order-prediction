import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import interpolate_missing, detrend, handle_holidays, week_scaling, avg_weeks, get_fourier_decomp

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# preprocessing
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

weekdays = train_df['date'].dt.dayofweek
weekdays = weekdays.apply(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
train_df['weekday'] = weekdays

weekdays = test_df['date'].dt.dayofweek
weekdays = weekdays.apply(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
test_df['weekday'] = weekdays

# Interpolation of missing data, while excluding days on which the warehouse is closed
train_df = interpolate_missing(train_df)

for warehouse in train_df['warehouse'].unique():
    week_length = 6 if warehouse in ['Munich_1', 'Frankfurt_1'] else 7

    warehouse_df_s = train_df[train_df['warehouse'] == warehouse]

    warehouse_df_s = interpolate_missing(warehouse_df_s)

    warehouse_df_hd, holiday_effects = handle_holidays(warehouse_df_s)

    warehouse_df_dt, trend_model = detrend(warehouse_df_hd)

    week_means = week_scaling(warehouse_df_dt, week_length=week_length)

    week_avgs = avg_weeks(warehouse_df_dt)
