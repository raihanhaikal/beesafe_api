#!/bin/bash

source ./venv/bin/activate
pip install -r mod.txt
gdown https://drive.google.com/uc?id=1lphpSnzlD3zAtQNpKu-91vKdXRcALGAX
mv models.zip models
unzip models/models.zip
docker-compose up --build -d