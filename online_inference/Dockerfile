FROM python:3.8-slim

MAINTAINER d3vyatk4ru 

ENV PATH_TO_MODEL="models/log_reg.pkl"

COPY init.sh requirements.txt app.py models/ src/ ./
RUN ./init.sh

WORKDIR .

CMD [ "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "9090"]