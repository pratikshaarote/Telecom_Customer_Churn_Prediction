import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import os
import warnings
warnings.filterwarnings('ignore')
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import datetime
import pickle as pk
import json
import mysql.connector

def get_data():
    mydb = mysql.connector.connect(
        host = "localhost",
        user = "root",
        password = "Pratiksha2501",
        database = "project"
        )
    mycursor = mydb.cursor()
        
    mycursor.execute("SELECT * FROM telco_customer_churn")
        
        # This SQL statement selects all data from the CUSTOMER table.
    result = mycursor.fetchall()

    df = pd.DataFrame(result,columns=['CustomerID','Count', 'Country', 'State', 'City','Zip_Code','Lat_Long','Longitude',
                    'Latitude','Gender','Senior Citizen','Partner','Dependents','Tenure Months','Phone Service','Multiple Lines','Internet Service',
                    'Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies','Contract',
                    'Paperless Billing','Payment Method','Monthly Charges','Total Charges','Churthn Label','Churn Value','Churn Score','CLTV','Churn Reason'])
    return df

def preprocess_data(df):
    df = df[['Gender', 'Senior Citizen',
           'Partner', 'Dependents', 'Tenure Months', 'Phone Service',
           'Multiple Lines', 'Internet Service', 'Online Security',
           'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
           'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method',
           'Monthly Charges', 'Total Charges']]

    df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})

    y_n_cols = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines','Online Security', 'Online Backup', 
        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Paperless Billing']
    df[y_n_cols] = df[y_n_cols].replace({'Yes':1, 'No':0}).replace(regex=r'No.*', value=0)

    df['Internet Service'] = df['Internet Service'].replace({'Fiber optic': 0, 'DSL': 1, 'No': 2})
    df['Contract'] = df['Contract'].replace({'Month-to-month': 0, 'Two year': 1, 'One year': 2})
    df['Payment Method'] = df['Payment Method'].replace({'Electronic check': 0,
     'Mailed check': 1,
     'Bank transfer (automatic)': 2,
     'Credit card (automatic)': 3})


    def adj_total_charges(df):
        try:
            if type(df['Total Charges']) in [int, float]:
                return float(df['Total Charges'])
            else:
                return df['Monthly Charges']
        except:
            return df['Monthly Charges']

    df['Total Charges'] = df.apply(adj_total_charges, axis = 1)
    df[['Monthly Charges', 'Total Charges']].astype(float)
    
    # scaler = pk.load(open('C:/Data science/Project/vscode_project/model_20230225/scaler.pkl', 'rb'))
    
    # df[['Monthly Charges', 'Total Charges']] = scaler.transform(df[['Monthly Charges', 'Total Charges']])
    return df 


def get_predictions(df):
    model = pk.load(open('C:/Data science/Project/vscode_project/Model_evaluation/model_20230301/model.pkl', 'rb'))
    return model.predict(df)


if __name__ == '__main__':
    df = get_data()
    df['Churn Prediction'] = get_predictions(preprocess_data(get_data()))
    df = df[df['Churn Prediction'] == 1]
    results = df[['CustomerID', 'Churn Prediction']].to_dict(orient='records')
    
    path = 'C:/Data science/Project/vscode_project/predictions'
    try:
        os.mkdir(path)
        with open(f"{path}/predictions_{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')}.json", "w") as f:
            json.dump(results, f)
    except:
        with open(f"{path}/predictions_{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')}.json", "w") as f:
            json.dump(results, f)
  