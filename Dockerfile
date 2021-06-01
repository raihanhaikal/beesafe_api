FROM python:3.8.6-slim

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# update command linux
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# install dependencies
COPY mod.txt .
RUN pip install -r mod.txt

# copy project
COPY . .

EXPOSE 8008:8000