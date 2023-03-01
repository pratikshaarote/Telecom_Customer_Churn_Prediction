import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import os
import warnings
warnings.filterwarnings('ignore')
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import datetime
import pickle as pk
import mysql.connector

def get_data():
    """fetch data from database and return dataframe"""
    mydb = mysql.connector.connect(
        host = "localhost",
        user = "root",
        password = "Pratiksha2501",
        database = "project"
        )
    mycursor = mydb.cursor()
        
    mycursor.execute("SELECT * FROM telco_customer_churn")
        

    result = mycursor.fetchall()

    df = pd.DataFrame(result,columns=['CustomerID','Count', 'Country', 'State', 'City','Zip_Code','Lat_Long','Longitude',
                    'Latitude','Gender','Senior Citizen','Partner','Dependents','Tenure Months','Phone Service','Multiple Lines','Internet Service',
                    'Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies','Contract',
                    'Paperless Billing','Payment Method','Monthly Charges','Total Charges','Churthn Label','Churn Value','Churn Score','CLTV','Churn Reason'])
    return df

def preprocess_data(df):
    """preprocessing data"""
    df = df[['Gender', 'Senior Citizen',
           'Partner', 'Dependents', 'Tenure Months', 'Phone Service',
           'Multiple Lines', 'Internet Service', 'Online Security',
           'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
           'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method',
           'Monthly Charges', 'Total Charges', 'Churn Value']]

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
    
    # global scaler
    # scaler = MinMaxScaler()
    # df[['Monthly Charges', 'Total Charges']] = scaler.fit_transform(df[['Monthly Charges', 'Total Charges']])
    return df 


def split_data(df):
    x = df.drop('Churn Value', axis = 1)
    y = df['Churn Value']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
    '''balancing the data'''
    sm_enn = SMOTEENN()
    x_train_res, y_train_res = sm_enn.fit_resample(x_train,y_train)
    x_train_sap, y_train_sap, x_test_sap, y_test_sap= train_test_split(x_train_res,y_train_res,test_size=0.2)

    return x_train_sap, y_train_sap, x_test_sap, y_test_sap


def train_model(x_train_sap, y_train_sap):
    """trains model on data and return model object"""
    Rfc_sampling = RandomForestClassifier(n_estimators=150,criterion='gini', max_depth=15, min_samples_leaf=10, min_samples_split=6)
    Rfc_sampling.fit(x_train_sap, y_train_sap)
    
    
    return Rfc_sampling

def store_model(x_train_sap, y_train_sap, x_test_sap, y_test_sap):  
    Rfc_sampling= train_model(x_train_sap, y_train_sap)
    path = f'C:/Data science/Project/vscode_project/Model_evaluation/model_{datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d")}'
    try:
        os.mkdir(path)
        # creating pickle file
        pk.dump(Rfc_sampling, open(f'{path}/model.pkl', 'wb'))

        with open(f'{path}/model_performance.txt', 'w') as f:
            content = f"""Model : {Rfc_sampling}\n
            {'*'*10} Train Results {'*'*10}\n
            {classification_report(y_train_sap, Rfc_sampling.predict(x_train_sap))}\n\n
            {'*'*10} Test Results {'*'*10}\n
            {classification_report(y_test_sap, Rfc_sampling.predict(x_test_sap))}"""
            f.write(content)
    except:
        pk.dump(Rfc_sampling, open(f'{path}/model.pkl', 'wb'))

        with open(f'{path}/model_performance.txt', 'w') as f:
            content = f"""Model : {Rfc_sampling}\n
            {'*'*10} Train Results {'*'*10}\n
            {classification_report(y_train_sap, Rfc_sampling.predict(x_train_sap))}\n\n
            {'*'*10} Test Results {'*'*10}\n
            {classification_report(y_test_sap, Rfc_sampling.predict(x_test_sap))}"""
            f.write(content)

        
        
if __name__ == '__main__':
    # calling above functions
    # get_data()
    # df = preprocess_data(get_data())
    x_train_sap, x_test_sap, y_train_sap, y_test_sap = split_data(preprocess_data(get_data()))
    train_model(x_train_sap, y_train_sap)
    store_model(x_train_sap, y_train_sap, x_test_sap, y_test_sap)
