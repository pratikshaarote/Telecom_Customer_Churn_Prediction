from flask import Flask, request, jsonify, app, url_for
import requests
# functions from make_prediction python file
from make_predictions import get_data, preprocess_data, get_predictions

app = Flask(__name__)
@app.route('/predict_churn', methods = ['GET'])
def make_pred():
    df = get_data()
    print(type(df))
    df['Churn Prediction'] = get_predictions(preprocess_data(get_data()))
    df = df[df['Churn Prediction'] == 1]
    results = df[['CustomerID', 'Churn Prediction']].to_dict(orient='records')
    return jsonify(results)

if __name__ == '__main__':
    app.run()              