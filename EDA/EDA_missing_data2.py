# Load the dataset
import pandas as pd

train_df = pd.read_csv('../data/train.csv')

# Convert date columns to datetime
train_df['date'] = pd.to_datetime(train_df['date'])

# print information about the Munich_1 and Frankfurt_1 warehouses
warehouse_df = train_df[train_df['warehouse'] == 'Prague_1']

# Add weekday column to the dataframe
weekdays = warehouse_df['date'].dt.dayofweek

# Make the weekday string
weekdays = weekdays.apply(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])

# set the weekday column in the warehouse_df to weekdays
warehouse_df['weekday'] = weekdays
