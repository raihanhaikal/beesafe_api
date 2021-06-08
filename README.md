<h1 align="center">
  <br>
  <a href=""><img src="https://github.com/raihanhaikal/beesafe_api/blob/main/img/logo.png" alt="beesafe" width="200"></a>
  <br>
  BeeSafe API
  <br>
</h1>

#### API for Bee Safe application
This is an API for BeeSafe, which is containerized using Docker and deployed to Google Cloud Compute Engine.
## Features

- Fetch predictions for sexual harassment category (Ogling, Groping, Catcalling, Safe)

  

## Documentation

1. Create the models using Tensorflow and save the models in Google Drive
2. Create the main.py to initialize the FastAPI and preprocess the input
3. Put all libraries and dependencies in requirements.txt using 
```bash 
  pip freeze > requirements.txt
```
4. Create the Dockerfile and docker-compose.yml
5. Run the virtual environment using the command and install all the libraries and dependencies in requirements.txt
```bash 
  source ./venv/bin/activate
  pip install -r requirements.txt
```
6. Run docker compose up
```bash 
  docker-compose up
```
7. Test the API locally using Postman or similar programs (make sure models are already downloaded if testing locally)
8. Push to Github repository
9. Create an instance in Google Cloud Compute Engine
- Region: Southeast Asia 2 - Jakarta
- Disk size: >= 40 GB
10. Open the ssh 
11. Clone the API repository
12. Download the models from Google Drive using gdown command
```bash 
  gdown [Google Drive link]
```
13. Run step 5 until 6
14. Test the API using Postman or similar apps
```bash
https://external-ip-address:port/predict
```


## Related

[BeeSafe](https://github.com/reynardets/BeeSafe)
 is the main repository

  
## Authors

- [Raihan Haikal](https://github.com/raihanhaikal) - Machine Learning
- [Sherryl Sindarto](https://github.com/sherrylsin20) - Machine Learning

  
