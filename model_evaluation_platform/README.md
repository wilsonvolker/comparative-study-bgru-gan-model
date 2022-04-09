# Model evaluation platform
In addition to the model training, an evaluation platform (web-app based) is also developed. 

[Front-end web-app (Next.js based)](https://github.com/wilsonvolker/comparative-study-bgru-gan-model/tree/main/model_evaluation_platform/web-app)
<br/>
[API (FastAPI based)](https://github.com/wilsonvolker/comparative-study-bgru-gan-model/tree/main/model_evaluation_platform/api)

<hr/>

## Run with docker
To start the services
```console
docker-compose up -d
```

To first rebuild then start the services
```console
docker-compose up -d --build
```

To list the files in the docker container
```console
docker-compose run model_evaluation_platform_api ls -laR .
```