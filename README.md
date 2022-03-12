# Twitter dialect classification

## Table of contents
- [Description](#Description)
- [Getting Started](#Getting)
  * [Dependencies](#Dependencies)
  * [Installing](#Installing)
  * [Executing program](#Executing)
- [Results](#Results)
- [Author](#Author)
- [Acknowledgments](#Acknowledgments)


## Description<a name="Description"/>

In this project I used the [QADI](https://arxiv.org/pdf/2005.06557.pdf) dataset to classify the different dialects of Arabic language. I used machine learning methods specifically logestic regression, linear SVM and naive bayes. I also tried to compare those methods with deep learning models namely LSTM and word embeddings. I did many my experiments locally on my machine. I deployed my project using Flask  
There are 4 main files in this project
* 4 `.py` scripts (`Data_fetching.py`, `Data_pre_processing.py`, `Model_training.py` and `app.py`) with the final results/code     

**There are also**    
* 2 Jupyter notebooks with detailed code of all the experiments that I did
* The dataset    
* There are also the Flask App's files like HTML and CSS files
* there is also a `dicts.py` file which is there just to help in predicting during Flask deployment. it contains a processing function to process the user's input text

## Getting Started<a name="Getting"/>
### Dependencies<a name="Dependencies"/>
* Anaconda is a must
* tensorflow
* flask
* farasapy (you need to install jave in order to work)
* PyArabic
* gensim

### Installing<a name="Installing"/>

* this code was run successfully on my windows machine.
* it's recommended to create a new anaconda environment with
```
conda create -n tf tensorflow
conda activate tf
```
* you need to install the dependencies
```
conda install pandas
conda install scikit-learn 
conda install -c anaconda flask
conda install -c anaconda gensim
conda install tensorflow
pip install farasapy
pip install PyArabic
```
* in order to train the models that require pretrained word embedding you need to download word embedding from 
1. [Mazajak](http://mazajak.inf.ed.ac.uk:8000/) specifically the CBOW words that were trained on 100M tweets
2. [AraVec](https://github.com/bakrianoo/aravec) specifically the Unigrams CBOW Models with vector size of 100

### Executing program<a name="Executing"/>

* you need to run `.py` scripts in order in the command line. 
* if you're intersted you can open the jupyter notebook for full detailed code/experiments 

## Results<a name="Results"/>
* **Results of deep learning on the validation set**

| Model name        | Accuracy           | F1 score  |
| ------------- |:-------------:| -----:|
| Embedding layer without lstm from scratch      | 0.523 | 0.494 |
| LSTM from scratch      | 0.455      | 0.399   | 
| Embedding layer with finetuned AraVec | 0.524      |    0.493 |
| Embedding layer with finetuned Mazajak | 0.526      |    0.497 |
| LSTM with fixed pretrained embedding Mazajak | 0.313      |    0.181 |
| LSTM with fixed pretrained embedding AraVec | 0.125      |    0.012 |
* **results of the machine learning models on the validation set**

| Model name        | Accuracy           | F1 score  |
| ------------- |:-------------:| -----:|
| uni-gram      | 0.512 | 0.478 |
| two-gram SVM      | 0.538      |   0.507 |
* **comparison of the deep learning and machine learning on the test set**

| Model name        | Accuracy           | F1 score  |
| ------------- |:-------------:| -----:|
| two gram SVM      | 0.5388 | 0.5072 |
| Embedding layer with finetuned Mazajak      | 0.5288      |   0.5024 |

## Author<a name="Author"/>
name: Bassel Ali Mahmoud   
email: basel_ebeed1@yahoo.com

## Acknowledgments<a name="Acknowledgments"/>
* [Mazajak](http://mazajak.inf.ed.ac.uk:8000/)
* [AraVec](https://github.com/bakrianoo/aravec)
* [farasa](https://farasa.qcri.org/)
* [farasapy](https://github.com/MagedSaeed/farasapy)
* [QADI](https://arxiv.org/pdf/2005.06557.pdf)
