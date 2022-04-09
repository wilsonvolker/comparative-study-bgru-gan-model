# A Comparative Study of BGRU and GAN for Stock Market Forecasting in dual regions
## - Codes
Run the codes in sequences: 0. Stock filtering, 1. Data fetching, 2, 3, ... n

## - Model evaluation platform
In addition to the model training, an evaluation platform (web-app based) is also developed. 

[Front-end web-app (Next.js based)](https://github.com/wilsonvolker/comparative-study-bgru-gan-model/tree/main/model_evaluation_platform/web-app)
<br/>
[API (FastAPI based)](https://github.com/wilsonvolker/comparative-study-bgru-gan-model/tree/main/model_evaluation_platform/api)

<hr/>

### To install packages
```bash
pip install -r /path/to/requirements.txt
# or
conda install --file /path/to/requirements.txt
```

### To export newly added package to requirement.txt
```console
pip install pipreqs
pipreqs /path/to/project --force
```
<b>** Remember to remove platform-based dependency, such as tensorflow-macos</b>
<br/>
<b>** Change `numpy==1.22.3` if its version is `< 1.20`<b>

### Other information
1. The files that contain "htgc" keywords, or with ".sh" extension, is to submit training jobs to CityU's High Throughput GPU Cluster 
   1. https://cslab.cs.cityu.edu.hk/services/high-throughput-gpu-cluster-3-htgc3

