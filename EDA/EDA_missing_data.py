import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks


def detect_periodicity(missing_dates):
    # Convert missing dates to a binary time series
    date_range = pd.date_range(start=missing_dates.min(), end=missing_dates.max(), freq='D')
    binary_series = pd.Series(0, index=date_range)
    binary_series.loc[missing_dates] = 1

    # Perform FFT
    fft_result = fft(binary_series.values)
    fft_freq = np.fft.fftfreq(len(binary_series), d=1)

    # Find peaks in the FFT magnitude spectrum
    peaks, _ = find_peaks(np.abs(fft_result), height=0.1 * len(binary_series))

    # Convert peak frequencies to periods
    periods = 1 / fft_freq[peaks]
    periods = periods[periods > 0]  # Remove negative frequencies

    return periods


# Load the dataset
train_df = pd.read_csv('../data/train.csv')

# Convert date columns to datetime
train_df['date'] = pd.to_datetime(train_df['date'])

for warehouse in train_df['warehouse'].unique():
    # Filter data for the current warehouse
    warehouse_df = train_df[train_df['warehouse'] == warehouse]

    # Create a complete date range
    date_range = pd.date_range(start=warehouse_df['date'].min(), end=warehouse_df['date'].max(), freq='D')

    # Identify missing dates
    missing_dates = date_range.difference(warehouse_df['date'])

    print(f"\nWarehouse: {warehouse}")
    print(f"Number of missing dates: {len(missing_dates)}")

    if len(missing_dates) > 0:
        # Detect periodicity
        periods = detect_periodicity(missing_dates)

        if len(periods) > 0:
            print("Detected periodicities (in days):")
            for period in sorted(periods):
                print(f"  {period:.2f}")
        else:
            print("No clear periodicity detected in missing dates.")

        # Plot missing dates
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(date_range, np.ones(len(date_range)), '|', color='blue', label='All dates')
        plt.plot(missing_dates, np.ones(len(missing_dates)), '|', color='red', label='Missing dates')
        plt.legend()
        plt.title(f'Missing Dates for {warehouse}')
        plt.xlabel('Date')

        # Plot FFT magnitude spectrum
        binary_series = pd.Series(0, index=date_range)
        binary_series.loc[missing_dates] = 1
        fft_result = fft(binary_series.values)
        fft_freq = np.fft.fftfreq(len(binary_series), d=1)

        # Convert frequency to period (in days)
        periods = 1 / fft_freq[1:len(fft_freq) // 2]
        magnitudes = np.abs(fft_result[1:len(fft_result) // 2])

        plt.subplot(2, 1, 2)
        plt.plot(periods, magnitudes)
        plt.title('FFT Magnitude Spectrum')
        plt.xlabel('Period (days)')
        plt.ylabel('Magnitude')
        plt.xscale('log')  # Use log scale for x-axis
        plt.xlim(1, len(binary_series))  # Limit x-axis from 1 day to the length of the series

        # Add vertical lines for common periods
        common_periods = [7, 30, 365.25]  # weekly, monthly, yearly
        for period in common_periods:
            plt.axvline(x=period, color='r', linestyle='--', alpha=0.5)
            plt.text(period, plt.ylim()[1], f'{period} days', rotation=90, va='top', ha='right', alpha=0.5)

        plt.tight_layout()
        plt.show()



