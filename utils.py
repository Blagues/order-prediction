import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

exclude_days_from_interpolate = {'Munch_1': ['Sunday'], 'Frankfurt_1': ['Sunday']}


def interpolate_missing(df):
    """
    Interpolate the missing data in the dataframe, respecting excluded days
    even with multiple consecutive missing days.
    """
    df_interpolated = df.copy()
    for warehouse in df['warehouse'].unique():
        exclude_days = exclude_days_from_interpolate.get(warehouse, [])
        warehouse_mask = df_interpolated['warehouse'] == warehouse
        for col in df_interpolated.select_dtypes(include=[np.number]).columns:
            # Create a helper column for interpolation
            df_interpolated.loc[warehouse_mask, 'helper'] = df_interpolated.loc[warehouse_mask, col]
            df_interpolated.loc[warehouse_mask & df_interpolated['weekday'].isin(exclude_days), 'helper'] = np.nan

            # Interpolate the helper column
            df_interpolated.loc[warehouse_mask, 'helper'] = df_interpolated.loc[warehouse_mask, 'helper'].interpolate()

            # Update the original column where it was null and not an excluded day
            mask = warehouse_mask & df_interpolated[col].isnull() & ~df_interpolated['weekday'].isin(exclude_days)
            df_interpolated.loc[mask, col] = df_interpolated.loc[mask, 'helper']

            # Drop the helper column
            df_interpolated.drop('helper', axis=1, inplace=True)

    return df_interpolated


def detrend(warehouse_df, plot=False):
    """
    Detrend the data using linear regression.
    """
    X = np.arange(len(warehouse_df)).reshape(-1, 1)
    model = LinearRegression().fit(X, warehouse_df['orders'])
    trend = model.predict(X)
    detrended = warehouse_df['orders'] - trend

    if plot:
        plt.figure(figsize=(12, 8))
        plt.plot(warehouse_df['orders'], label='Original data')
        plt.plot(detrended, label='Detrended data')
        plt.show()

    detrended_df = warehouse_df.copy()
    detrended_df['detrended_orders'] = detrended

    return detrended_df, model


def handle_holidays(warehouse_df):
    """
    For each day that is a holiday, the day before a holiday, or the two days before Christmas Eve,
    we calculate the average increase with respect to the previous non-holiday day and the next non-holiday day.
    We do this for each distinct 'holiday_name' and store the average effect in a dictionary.
    Then we remove the effect by dividing the orders by the holiday factor.
    """
    holiday_effects = {}
    df = warehouse_df.copy()

    # Mark days before holidays and special days before Christmas Eve
    df['day_before_holiday'] = df['holiday'].shift(-1).fillna(0).astype(bool)
    df['holiday'] = df['holiday'].astype(bool)
    df['holiday_or_before'] = df['holiday'] | df['day_before_holiday']

    # Mark two days before Christmas Eve
    christmas_eve = df[df['holiday_name'] == 'Christmas Eve'].index
    if not christmas_eve.empty:
        df.loc[christmas_eve - 1, 'holiday_or_before'] = True
        df.loc[christmas_eve - 2, 'holiday_or_before'] = True

    # Calculate holiday effects
    holiday_days = df[df['holiday_or_before']]

    for idx in holiday_days.index:
        if df.loc[idx, 'holiday']:
            holiday_name = df.loc[idx, 'holiday_name'] if 'holiday_name' in df.columns else 'Unnamed Holiday'
        elif idx in christmas_eve - 1:
            holiday_name = "Day before Christmas Eve"
        elif idx in christmas_eve - 2:
            holiday_name = "Two days before Christmas Eve"
        else:
            holiday_name = f"Day before {df.loc[idx + 1, 'holiday_name'] if 'holiday_name' in df.columns else 'Unnamed Holiday'}"

        # Find the previous non-holiday day
        prev_non_holiday_idx = df.loc[:idx][~df['holiday_or_before']].index[-1] if any(
            ~df.loc[:idx]['holiday_or_before']) else None

        # Find the next non-holiday day
        next_non_holiday_idx = df.loc[idx:][~df['holiday_or_before']].index[0] if any(
            ~df.loc[idx:]['holiday_or_before']) else None

        if prev_non_holiday_idx is not None and next_non_holiday_idx is not None:
            prev_non_holiday_orders = df.loc[prev_non_holiday_idx, 'orders']
            next_non_holiday_orders = df.loc[next_non_holiday_idx, 'orders']
            holiday_orders = df.loc[idx, 'orders']

            # Calculate effect relative to previous non-holiday day
            prev_effect = holiday_orders / prev_non_holiday_orders if prev_non_holiday_orders != 0 else 1

            # Calculate effect relative to next non-holiday day
            next_effect = holiday_orders / next_non_holiday_orders if next_non_holiday_orders != 0 else 1

            # Average the two effects
            effect = (prev_effect + next_effect) / 2

            if holiday_name not in holiday_effects:
                holiday_effects[holiday_name] = []
            holiday_effects[holiday_name].append(effect)

    # Calculate average effect for each holiday
    for holiday, effects in holiday_effects.items():
        holiday_effects[holiday] = np.mean(effects)

    # Remove holiday effects
    for idx in holiday_days.index:
        if df.loc[idx, 'holiday']:
            holiday_name = df.loc[idx, 'holiday_name'] if 'holiday_name' in df.columns else 'Unnamed Holiday'
        elif idx in christmas_eve - 1:
            holiday_name = "Day before Christmas Eve"
        elif idx in christmas_eve - 2:
            holiday_name = "Two days before Christmas Eve"
        else:
            holiday_name = f"Day before {df.loc[idx + 1, 'holiday_name'] if 'holiday_name' in df.columns else 'Unnamed Holiday'}"
        if holiday_name in holiday_effects:
            df.loc[idx, 'orders'] = df.loc[idx, 'orders'] / holiday_effects[holiday_name]

    # Add a new column with the holiday factors
    df['holiday_factor'] = 1  # Default for non-holiday days
    for idx in holiday_days.index:
        if df.loc[idx, 'holiday']:
            holiday_name = df.loc[idx, 'holiday_name'] if 'holiday_name' in df.columns else 'Unnamed Holiday'
        elif idx in christmas_eve - 1:
            holiday_name = "Day before Christmas Eve"
        elif idx in christmas_eve - 2:
            holiday_name = "Two days before Christmas Eve"
        else:
            holiday_name = f"Day before {df.loc[idx + 1, 'holiday_name'] if 'holiday_name' in df.columns else 'Unnamed Holiday'}"
        if holiday_name in holiday_effects:
            df.loc[idx, 'holiday_factor'] = holiday_effects[holiday_name]

    # Clean up temporary columns
    df = df.drop(['day_before_holiday', 'holiday_or_before'], axis=1)

    return df, holiday_effects


import pandas as pd
import numpy as np


def week_scaling(warehouse_df, week_length=7):
    total_week_avgs = []

    this_week = []

    for idx, row in warehouse_df.iterrows():

        if row['weekday'] == 'Monday':
            if len(this_week) == week_length:
                week_avg = np.mean(this_week)
                total_week_avgs.append(this_week / week_avg)

            this_week = []

        this_week.append(row['detrended_orders'])

    final_means = np.mean(total_week_avgs, axis=0)

    return final_means


def avg_weeks(warehouse_df):
    """
    Returns a list that contains the average orders for each day of the week, considering distinct years.
    """

    # add year and week columns
    warehouse_df['year'] = warehouse_df['date'].dt.year
    warehouse_df['week'] = warehouse_df['date'].dt.isocalendar().week

    # group by year and week, then take the mean of the orders
    weekly_means = warehouse_df.groupby(['year', 'week'])['detrended_orders'].mean()

    return weekly_means.to_list()


import numpy as np


def get_fourier_decomp(data, n_components, n_future_days):
    """
    Get the Fourier decomposition model for the given data.
    Also calculates the Fourier decomposition for the given number of future days.

    Parameters:
        data (np.ndarray): The input time series data.
        n_components (int): The number of Fourier components to keep.
        n_future_days (int): The number of future days to predict.

    Returns:
        tuple: A tuple containing the reconstructed signal for the existing data and the reconstructed signal for the future data.
    """
    n = len(data)

    # Fourier transform
    fft_result = np.fft.fft(data)
    magnitudes = np.abs(fft_result)
    top_indices = np.argsort(magnitudes)[-n_components:]
    filtered_fft = np.zeros_like(fft_result)
    filtered_fft[top_indices] = fft_result[top_indices]

    # Inverse Fourier transform to get the reconstructed signal
    reconstructed = np.fft.ifft(filtered_fft).real

    # Generate future Fourier series
    future_time = np.arange(n, n + n_future_days)
    future_reconstructed = np.zeros(n_future_days)

    for k in top_indices:
        amplitude = np.abs(fft_result[k]) / n
        phase = np.angle(fft_result[k])
        frequency = 2 * np.pi * k / n

        future_reconstructed += amplitude * np.cos(frequency * future_time + phase)

    # Combine current and future reconstructions for continuity
    total_reconstructed = np.concatenate((reconstructed, future_reconstructed))

    return reconstructed, future_reconstructed, total_reconstructed

