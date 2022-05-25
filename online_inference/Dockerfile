FROM python:3.8.10-slim

MAINTAINER d3vyatk4ru 

ENV PATH_TO_MODEL="models/log_reg.pkl"

COPY requirements.txt\
     app.py\
     init.sh ./

COPY src /src
COPY  models ./models

RUN ./init.sh

WORKDIR .

EXPOSE 9090

CMD [ "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "9090"]