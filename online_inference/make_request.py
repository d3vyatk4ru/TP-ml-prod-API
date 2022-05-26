import pandas as pd
import requests
import os

DATA_PATH = os.path.abspath('online_inference\\data\\heart_cleveland_upload.csv')
TARGET = 'condition'


if __name__ == '__main__':

    data = pd.read_csv(DATA_PATH).drop(columns=TARGET)

    request_params = {
        'data': data.values.tolist(),
        'features': data.columns.tolist(),
    }

    response = requests.get(
        'http://0.0.0.0:9090/predict',
        json=request_params,
    )

    print(response.status_code)
    print(response.json())
    