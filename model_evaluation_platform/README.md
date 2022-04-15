# Model evaluation platform
In addition to the model training, an evaluation platform (web-app based) is also developed. 

[Front-end web-app (Next.js based)](web-app)
<br/>
[API (FastAPI based)](api)

<hr/>

## User Guide
[Model evaluation platform - User Guide](USER_GUIDE.md)

<hr/>

## Deployment Guide of the Model Evaluation Platform

### Suggested OS for this deployment guide
Ubuntu 18.04.6 LTS

### Prerequisites
1. [Docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04)
2. [Docker-compose](https://docs.docker.com/compose/install/)
3. Python 3.9.7
4. Tensorflow 2.8.0

## Deploying Frontend + Backend
1. Ensure the project is cloned from github <br/>
```console
git clone -b master https://github.com/wilsonvolker/comparative-study-bgru-gan-model.git
```
2. Ensure you are inside the model_evaluation_platform directory
```console
cd {path_to_project}/model_evaluation_platform
```
3. Update the <b>environment variables (args for frontend)</b> in `docker-compose.yml`
###### model_evaluation_platform_frontend:
```console
- NEXT_PUBLIC_EVALUATE_URL={ADD_YOUR_SERVER_URL_HERE}/evaluate
- NEXT_PUBLIC_DEFAULT_STOCKS_URL={ADD_YOUR_SERVER_URL_HERE}/default_stocks
```
###### model_evaluation_platform_api:
```console
- FRONT_END_URL={ADD_YOUR_SERVER_URL_HERE}
```
4. Run the following command to start the build process in docker:
##### To start the services
```console
docker-compose up -d
```

##### To first rebuild then start the services
```console
docker-compose up -d --build
```
5. Visit your application here:
<br/>
<b>Frontend:</b> http://{YOUR_SERVER_URL_HERE}/
<br/>
<b>Backend:</b> http://{YOUR_SERVER_URL_HERE}:8000/

<hr/>

### Some other useful command for docker container debugging

##### To list the files in the docker container
```console
docker-compose run model_evaluation_platform_api ls -laR .
```

##### To enter the bash (ssh/cli) of the docker container
###### For Frontend
```console
docker exec -it {container_id} /bin/sh
```
###### For backend API
```console
docker exec -it {container_id} bash
```
