import pandas as pd
import numpy as np
import requests

URL = 'https://recruitment.aimtechnologies.co/ai-tasks'
df = pd.read_csv('dialect_dataset.csv', dtype=str)

batches = df.shape[0]//1000
size = 1000
data_id = [] 
data_tweets = []
for batch in range(batches+1):
    #print("===========================")
    ids = np.array(df['id'][size*batch:size*batch+size]).tolist()
    data_id.extend(ids)
    req = requests.post(URL, json= ids)
    data_tweets.extend(req.json().values())

ids = np.array(df['id'][size*(batch+1):]).tolist()
data_id.extend(ids)
req = requests.post(URL, json= ids)
data_tweets.extend(req.json().values())
dataset = pd.DataFrame({'dialect': df['dialect'], 'tweet': data_tweets})

dataset.to_csv('dataset_with_tweets.csv', index=False)