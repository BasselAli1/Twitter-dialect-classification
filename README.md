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

In this project I used the [QADI](https://arxiv.org/pdf/2005.06557.pdf) dataset to classify the different dialects of Arabic language. I used machine learning methods specifically logestic regression, linear SVM and naive bayes. I also tried to compare those methods with deep learning models namely LSTM and word embeddings. I did many my experiments locally on my machine.  
There are 4 main types of files in this project
* 4 `.py` scripts (Data fetching, Data pre-processing, Model Training and Deployment) with the final results/code
* 2 Jupyter notebooks with detailed code of all the experiments that I did
* 2 pre-trained word embedding files that I downloaded
1. [Mazajak](http://mazajak.inf.ed.ac.uk:8000/) specifically the CBOW words that were trained on 100M tweets
2. [AraVec](https://github.com/bakrianoo/aravec) specifically the Unigrams CBOW Models with vector size of 100
* The dataset before preprocessing (`dataset_with_tweets.csv`) and after preprocessing (`preprocessed_data.csv`)   
Threre are also the Flask App's files like HTML and CSS files

## Getting Started<a name="Getting"/>
### Dependencies<a name="Dependencies"/>
* Anaconda is a must
* tensorflow
* flask
* farasapy
* PyArabic
* gensim

### Installing<a name="Installing"/>

* this code was run successfully on my windows machine.
* it's recommended to create a new anaconda environment with
```
conda create -n tf tensorflow
conda activate tf
```
* you can install the dependencies yourself or run the `dependencies.bat` file from the anaconda command-line

### Executing program<a name="Executing"/>

* you can simply run any `.py` script in the command line
* if you're intersted you can open the jupyter notebook for full detailed code/experiments 

## Results<a name="Results"/>

| Model name        | Accuracy           | F1 score  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |
| zebra stripes | are neat      |    $1 |
| zebra stripes | are neat      |    $1 |
| zebra stripes | are neat      |    $1 |

| Model name        | Accuracy           | F1 score  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |

| Model name        | Accuracy           | F1 score  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |

## Author<a name="Author"/>
name: Bassel Ali Mahmoud   
email: basel_ebeed1@yahoo.com

## Acknowledgments<a name="Acknowledgments"/>
* [Mazajak](http://mazajak.inf.ed.ac.uk:8000/)
* [AraVec](https://github.com/bakrianoo/aravec)
* [farasa](https://farasa.qcri.org/)
* [farasapy](https://github.com/MagedSaeed/farasapy)
* [QADI](https://arxiv.org/pdf/2005.06557.pdf)