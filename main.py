import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Convert date columns to datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


# Function to interpolate missing data
def interpolate_missing(df):
    # Create a copy of the dataframe to avoid modifying the original
    df_interpolated = df.copy()

    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Interpolate only numeric columns
    df_interpolated[numeric_columns] = df_interpolated[numeric_columns].interpolate()

    return df_interpolated


# Function to detrend the data
def detrend(data):
    X = np.arange(len(data)).reshape(-1, 1)
    model = LinearRegression().fit(X, data)
    trend = model.predict(X)
    detrended = data - trend
    return detrended, model


def handle_holidays(df):
    holiday_effects = {}
    for holiday in df[df['holiday'] == 1]['holiday_name'].unique():
        holiday_data = df[df['holiday_name'] == holiday]
        non_holiday_data = df[(df['holiday'] == 0) & (df['date'] < holiday_data['date'].min())]
        if len(non_holiday_data) > 0:
            last_non_holiday = non_holiday_data.iloc[-1]
            effect = holiday_data['orders'].mean() / last_non_holiday['orders']
            holiday_effects[holiday] = effect

    df['holiday_factor'] = 1.0  # Initialize as float
    for holiday, effect in holiday_effects.items():
        df.loc[df['holiday_name'] == holiday, 'holiday_factor'] = effect

    df['adjusted_orders'] = df['orders'].astype(float) / df['holiday_factor']
    return df, holiday_effects


def week_scaling(df):
    df['week'] = df['date'].dt.isocalendar().week
    df['weekday'] = df['date'].dt.dayofweek

    weekly_means = df.groupby('week')['adjusted_orders'].mean()
    df['weekly_mean'] = df['week'].map(weekly_means)

    df['day_deviation'] = df['adjusted_orders'].astype(float) / df['weekly_mean']

    average_day_deviations = df.groupby('weekday')['day_deviation'].mean()

    return df, average_day_deviations


# Function for Fourier decomposition
def fourier_decomposition(data, n_components):
    fft_result = fft(data)
    magnitudes = np.abs(fft_result)
    top_indices = np.argsort(magnitudes)[-n_components:]
    filtered_fft = np.zeros_like(fft_result)
    filtered_fft[top_indices] = fft_result[top_indices]
    return ifft(filtered_fft).real


def process_warehouse(df, n_test=21, n_components=2):
    # Split into train and test
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]

    # Interpolate missing data
    train = interpolate_missing(train)

    # Visualize original and interpolated data
    plt.figure(figsize=(15, 5))
    plt.plot(train['date'], train['orders'], label='Original')
    plt.plot(train['date'], train['orders'].interpolate(), label='Interpolated')
    plt.title('Original vs Interpolated Data')
    plt.legend()
    plt.show()

    # Detrend
    detrended, trend_model = detrend(train['orders'].values)

    # Visualize detrending
    plt.figure(figsize=(15, 5))
    plt.plot(train['date'], train['orders'], label='Original')
    plt.plot(train['date'], trend_model.predict(np.arange(len(train)).reshape(-1, 1)), label='Trend')
    plt.plot(train['date'], detrended, label='Detrended')
    plt.title('Detrending Visualization')
    plt.legend()
    plt.show()

    # Handle holidays
    train, holiday_effects = handle_holidays(train)

    # Visualize holiday effects
    plt.figure(figsize=(15, 5))
    plt.plot(train['date'], train['orders'], label='Original')
    plt.plot(train['date'], train['adjusted_orders'], label='Holiday Adjusted')
    plt.title('Holiday Effect Adjustment')
    plt.legend()
    plt.show()

    # Week scaling
    train, day_deviations = week_scaling(train)

    # Visualize week scaling
    plt.figure(figsize=(15, 5))
    plt.bar(range(7), day_deviations.values)
    plt.title('Average Day Deviations')
    plt.xlabel('Day of Week')
    plt.ylabel('Deviation')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()

    # Fourier decomposition
    fourier_series = fourier_decomposition(train['adjusted_orders'].values, n_components)

    # Visualize Fourier decomposition
    plt.figure(figsize=(15, 5))
    plt.plot(train['date'], train['adjusted_orders'], label='Adjusted Data')
    plt.plot(train['date'], fourier_series, label='Fourier Approximation')
    plt.title(f'Fourier Decomposition (n_components={n_components})')
    plt.legend()
    plt.show()

    # Predict for test period
    test_predictions = []
    for _, row in test.iterrows():
        # Start with the Fourier prediction
        pred = fourier_series[len(train) % len(fourier_series)]

        # Apply week scaling
        pred *= day_deviations[row['date'].dayofweek]

        # Apply holiday effect if applicable
        if row['holiday'] == 1:
            pred *= holiday_effects.get(row['holiday_name'], 1)

        # Add trend
        pred += trend_model.predict([[len(train)]])[0]

        test_predictions.append(pred)
        train = pd.concat([train, row.to_frame().T], ignore_index=True)

    # Visualize final prediction
    plt.figure(figsize=(15, 5))
    plt.plot(test['date'], test['orders'], label='Actual')
    plt.plot(test['date'], test_predictions, label='Predicted')
    plt.title('Final Prediction vs Actual')
    plt.legend()
    plt.show()

    return test_predictions, test['orders'].values


# Process all warehouses
results = {}
for warehouse in train_df['warehouse'].unique():
    print(f"Processing Warehouse {warehouse}")
    warehouse_data = train_df[train_df['warehouse'] == warehouse].sort_values('date')
    predictions, actuals = process_warehouse(warehouse_data)
    results[warehouse] = {'predictions': predictions, 'actuals': actuals}
    print("\n" + "=" * 50 + "\n")

# Visualization
n_warehouses = len(results)
n_cols = 3  # You can adjust this to change the number of columns in the plot
n_rows = (n_warehouses - 1) // n_cols + 1

plt.figure(figsize=(20, 6 * n_rows))
for i, (warehouse, data) in enumerate(results.items(), 1):
    plt.subplot(n_rows, n_cols, i)
    plt.plot(data['actuals'], label='Actual')
    plt.plot(data['predictions'], label='Predicted')
    plt.title(f'Warehouse {warehouse}')
    plt.legend()
plt.tight_layout()
plt.show()

# Calculate RMSE for each warehouse
for warehouse, data in results.items():
    rmse = np.sqrt(np.mean((data['predictions'] - data['actuals']) ** 2))
    print(f'RMSE for Warehouse {warehouse}: {rmse}')