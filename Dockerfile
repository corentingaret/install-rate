FROM python:3.11.8-slim

RUN mkdir -p /code/job /code/data
WORKDIR /code/job

RUN pip install --upgrade pip && pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /code/job/
RUN poetry install --without ci

COPY src/job /code/job
