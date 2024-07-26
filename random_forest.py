from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='client/build')
CORS(app)

# Load the data
files = {
    "D19": "Data Files/D19.csv",
    "D21": "Data Files/D21.csv",
    "GP21": "Data Files/GP21.csv",
    "MSE21": "Data Files/MSE21.csv",
    "NP21": "Data Files/NP21.csv",
    "SRM22": "Data Files/SRM22.csv",
    "SRM23": "Data Files/SRM23.csv"
}

event_dates = {
    'D19': '2019-11-19',
    'D21': '2021-12-09',
    'GP21': '2021-04-22',
    'MSE21': '2021-03-24',
    'NP21': '2021-11-09',
    'SRM22': '2022-06-15',
    'SRM23': '2023-06-08'
}

dataframes = {key: pd.read_csv(file) for key, file in files.items()}

# Function to clean the dataframe
def clean_dataframe(df):
    df = df.drop(columns=['BookingReference', 'Attended'])
    df = df[df['Attendee Status'] != 'Cancelled']
    df['Attendee Status'] = 'Attending'
    return df

cleaned_dataframes = {key: clean_dataframe(df) for key, df in dataframes.items()}

for key, df in cleaned_dataframes.items():
    df['Event Date'] = event_dates[key]

# Combine all the cleaned dataframes into one dataframe
combined_df = pd.concat(cleaned_dataframes.values(), ignore_index=True)

# Check for missing values
missing_values = combined_df.isnull().sum()

# Convert date columns to datetime
combined_df['Created Date'] = pd.to_datetime(combined_df['Created Date'], dayfirst=True)
combined_df['Event Date'] = pd.to_datetime(combined_df['Event Date'])

# Calculate days until event and other features
combined_df['Days Until Event'] = (combined_df['Event Date'] - combined_df['Created Date']).dt.days
combined_df['Weekday'] = combined_df['Created Date'].dt.day_name()
combined_df['Week of Year'] = combined_df['Created Date'].dt.isocalendar().week
combined_df['Month'] = combined_df['Created Date'].dt.month
combined_df = combined_df.sort_values(by=['Event Date', 'Created Date'])
combined_df['Cumulative Registrations'] = combined_df.groupby('Event Date').cumcount() + 1

# Encode the 'Weekday' feature using one-hot encoding
encoded_df = pd.get_dummies(combined_df, columns=['Weekday'])

# Calculate daily registrations
combined_df['Daily Registrations'] = combined_df.groupby(['Event Date', 'Created Date']).cumcount() + 1

# Prepare the dataset for modeling
modeling_df = combined_df[['Event Date', 'Created Date', 'Days Until Event', 'Cumulative Registrations', 'Daily Registrations']]

# Model training
features = ['Days Until Event', 'Cumulative Registrations']
target = 'Daily Registrations'
X = modeling_df[features]
y = modeling_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Calculate the standard deviation of the residuals
residuals = y_test - y_pred
std_residuals = np.std(residuals)

# Function to predict future daily and cumulative registrations with confidence intervals
def predict_future_registrations_with_ci(current_registrations, event_date_str, std_residuals, confidence_level=1.96):
    event_date = pd.to_datetime(event_date_str)
    days_until_event = (event_date - pd.Timestamp.now()).days
    future_dates = [pd.Timestamp.now() + timedelta(days=i) for i in range(days_until_event + 1)]
    daily_predictions = []
    cumulative_predictions = []
    cumulative_registrations = current_registrations
    for days_left in range(days_until_event, -1, -1):
        input_data = pd.DataFrame({
            'Days Until Event': [days_left],
            'Cumulative Registrations': [cumulative_registrations]
        })
        predicted_daily_registrations = rf_model.predict(input_data)[0]
        daily_predictions.append(predicted_daily_registrations)
        cumulative_registrations += predicted_daily_registrations
        cumulative_predictions.append(cumulative_registrations)
    # Calculate confidence intervals
    lower_bound = np.array(cumulative_predictions) - (confidence_level * std_residuals)
    upper_bound = np.array(cumulative_predictions) + (confidence_level * std_residuals)
    # Calculate percentage range
    percentage_range = (confidence_level * std_residuals) / cumulative_predictions[-1] * 100
    return future_dates, daily_predictions, cumulative_predictions[-1], lower_bound[-1], upper_bound[-1], percentage_range

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    event_date = data.get('event_date')
    current_registrations = data.get('current_registrations')

    if not current_registrations or not event_date:
        return jsonify({'error': 'Invalid input data'}), 400

    try:
        future_dates, daily_predictions, final_predicted_registrations, lower_bound, upper_bound, percentage_range = predict_future_registrations_with_ci(current_registrations, event_date, std_residuals)
        
        result = {
            'final_predicted_registrations': final_predicted_registrations,
            'confidence_interval': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'percentage_range': percentage_range
        }
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
