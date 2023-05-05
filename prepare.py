# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def prepare_data(training_data, new_data):
    cpy_data = new_data.copy()
    cpy_training = training_data.copy()
    extract_features(cpy_data)
    extract_features(cpy_training)
    normelize_data(cpy_data, cpy_training)
    return cpy_data


def extract_features(data):
    extract_blood_type(data)
    extract_symptoms(data)
    change_date_to_int(data)
    change_sex_to_int(data)
    extract_location(data)
    reorder_risk_and_spread(data)


def extract_blood_type(data):
    low_risk = ["A+", "A-"]
    mid_risk = ["B+", "B-", "AB+", "AB-"]
    high_risk = ["O+", "O-"]
    data.loc[data['blood_type'].isin(low_risk), 'blood_type'] = -1
    data.loc[data['blood_type'].isin(mid_risk), 'blood_type'] = 0
    data.loc[data['blood_type'].isin(high_risk), 'blood_type'] = 1
    data['blood_group'] = data['blood_type'].astype(int)
    data.drop('blood_type', axis=1, inplace=True)


def extract_symptoms(data):
    symptoms = data["symptoms"].str.split(";", expand=True).stack().unique()
    for symptom in symptoms:
        data[symptom] = data["symptoms"].str.contains(symptom).fillna(0).replace({
            True: 1, False: -1}).astype(int)

    data.drop(columns=["symptoms"], inplace=True)


def change_sex_to_int(data):
    data.loc[data['sex'] == 'F', 'sex'] = -1
    data.loc[data['sex'] == 'M', 'sex'] = 1


def change_date_to_int(data):
    # Convert the 'date' column to a datetime column
    data['pcr_date'] = pd.to_datetime(data['pcr_date'])

    # Extract the integer values for the dates
    data['pcr_date'] = (data['pcr_date'] -
                        pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')


def extract_location(data):
    # Define a function to extract the first and second values from the split arrays
    def extract_values(arr):
        arr = arr[1:-1].split(',')
        x = float(arr[0].replace("'", '').replace(' ', ''))
        y = float(arr[1].replace("'", '').replace(' ', ''))
        return x, y

    data[['x_location', 'y_location']] = data['current_location'].apply(
        lambda arr: pd.Series(extract_values(arr)))

    data.drop(columns=['current_location'], inplace=True)


def reorder_risk_and_spread(data):
    spread = data.pop('spread')
    data['spread'] = spread
    risk = data.pop('risk')
    data['risk'] = risk


def normelize_data(data, training):
    min_max_columns = ['patient_id', 'sport_activity', 'pcr_date', 'PCR_01',
                       'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05', 'PCR_07',
                       'PCR_09', 'PCR_10']
    standard_columns = ['age', 'weight', 'num_of_siblings', 'happiness_score',
                        'household_income', 'conversations_per_day', 'sugar_levels',
                        'PCR_06', 'PCR_08', 'x_location', 'y_location']
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    standard_scaler = StandardScaler()
    for label in min_max_columns:
        data[label] = min_max_scaler.fit_transform(data[label])
        training[label] = min_max_scaler.transform(training[label])

    for label in standard_columns:
        data[label] = standard_scaler.fit_transform(
            data[label].values.reshape(-1, 1))
        training[label] = standard_scaler.transform(
            training[label].values.reshape(-1, 1))
