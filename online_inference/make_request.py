import pandas as pd
import numpy as np
import requests
import os

DATA_PATH = os.path.abspath(os.path.join('data', 'heart_cleveland_upload.csv'))
TARGET = 'condition'


if __name__ == '__main__':

    data = pd.read_csv(DATA_PATH).drop(columns=TARGET)
    request_features = list(data.columns)

    for i, _ in enumerate(data):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]

        print(request_data)

        response = requests.get(
            'http://0.0.0.0:9090/predict',
            json={'data': [request_data], 'features': request_features},
        )
        
        print(response.status_code)
        print(response.json())
    