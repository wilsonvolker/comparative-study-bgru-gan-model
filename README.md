# A Comparative Study of BGRU and GAN for Stock Market Forecasting in dual regions
![Project Screenshot - A Comparative Study of BGRU and GAN for Stock Market Forecasting in dual regions](diagrams/project%20screenshot.png)
### Objectives
This project will conduct a comparative study to evaluate the BGRU and GAN models on predicting the stock market trend in Hong Kong and the United States market. The following objectives will thus be established. 

1. To evaluate and compare the models’ daily stock close price prediction accuracy of the respective market. 

2. To evaluate whether the transfer learning can also be applied to the trained Hong Kong market models and United States market models by evaluating the respective prediction result of forecasting the opposites’ daily close price of the selected stocks. 

## Codes
Run the codes in sequences: 
<br>0. Stock filtering, 
<br>1. Data fetching, 
<br>2, 3, ... n

## Model evaluation platform
In addition to the model training, an evaluation platform (web-app based) is also developed. 

[Front-end web-app (Next.js based)](model_evaluation_platform/web-app)
<br/>
[API (FastAPI based)](model_evaluation_platform/api)

![Model Evaluation Platform Frontend - Input Form](diagrams/Model%20Evaluation%20Platform%20Frontend%20-%20Input%20form.png)
![Model Evaluation Platform Frontend - Result Page](diagrams/Model%20Evaluation%20Platform%20Frontend%20-%20Result%20page.png)

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

