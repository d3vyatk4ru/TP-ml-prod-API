import pickle
import os
import logging
from typing import NoReturn, List

from fastapi import FastAPI
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd

from src.respose import PredictResponse


MODEL = None
TRANSFORMER = None


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]:  [%(asctime)s] %(name)s  %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger('ml-service')

app = FastAPI()

@app.get('/')
async def start():
    """ Print string on start service """
    return 'Hello, bro!!!! DLKSLDA:D:KASDK:ASKDASLKDLA:S"DKSAAKASDSAJDAS"DASKDNKSANDKLSADJ"KSASAJDK'


@app.on_event('startup')
async def load_model() -> NoReturn:
    """ Load ML model """

    global MODEL

    # model_path = os.getenv('models/model.pkl')
    model_path = os.path.abspath('models\log_reg.pkl')

    logger.info('Model %s loading...', model_path)

    logger.info('Model %s successful loaded!', model_path)

    MODEL = load_pickle(model_path)


@app.get('/health')
def health() -> bool:
    """ Reaturn status of model """
    logger.info('Check health model')
    return 200 if not (MODEL is None) else 400


def make_predict(
    data: List,
    features: List[str],
    model: Pipeline,
) -> List[PredictResponse]:
    """ Make pridict """
    data = pd.DataFrame(data, columns=features)

    n_row = [i for i, _ in enumerate(data)]
    preds = model.predict(data)

    return [
        PredictResponse(id=index, target=target) for index, target in zip(n_row, preds)
    ]


@app.get('/predict')
def predict(request) -> List[PredictResponse]:
    """ Predict for model. Call the make_predict function """
    if not health():
        logger.error('Model is not health!')
        raise Exception("Can't make predict: model is not health")

    return make_predict(request.data, request.features, MODEL)


def load_pickle(path: str) -> Pipeline:
    """ Read object from pkl """

    with open(path, 'rb') as some_object:
        return pickle.load(some_object)


if __name__ == '__main__':
    uvicorn.main('app:app', host="0.0.0.0", port=os.getenv('PORT', 9090))
    print(pd.__version__)
