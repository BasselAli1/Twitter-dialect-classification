import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU,Dense, Dropout, SpatialDropout1D, GlobalAveragePooling1D, LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline
from joblib import dump
import tensorflow
import gensim
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


dataset = pd.read_csv("preprocessed_data.csv")

# removing one bad data point where the entire tweet is english
dataset = dataset.drop(dataset['tweet'][dataset['pure_tweet'].isnull()].index)

dataset['dialect_number'] = dataset['dialect'].factorize()[0]
outputs = dict(zip(dataset['dialect_number'], dataset['dialect']))

# splitting the data into training, validation test
X_train, X_other, y_train, y_other = train_test_split(dataset, dataset['dialect_number'],test_size = 0.2, random_state =0)
X_val, X_test, y_val, y_test = train_test_split(X_other, y_other,test_size = 0.5, random_state =0)

# deep learning
#==============
# tokenize the training and va
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train['pure_tweet'])
encoded_docs = tokenizer.texts_to_sequences(X_train['pure_tweet'])
y= to_categorical(y_train,num_classes=18)
# padding training data
padded_sequence = pad_sequences(encoded_docs, maxlen=60, padding='post')

# tokenize the validation data
y_val_categorical= to_categorical(y_val,num_classes=18)
val_tweets = tokenizer.texts_to_sequences(X_val['pure_tweet'])
# padding validation data
val_padded_sequence = pad_sequences(val_tweets, maxlen=60)

# vocabulary size
vocab_size = len(tokenizer.word_index)+1

# Loading the Mazajak Pretrained word embedding
embeddings_Mazajak = gensim.models.KeyedVectors.load_word2vec_format('cbow_100.bin',binary=True,unicode_errors='ignore')

embedding_matrix_Mazajak = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    try:
        embedding_vector = embeddings_Mazajak[word]
        if embedding_vector is not None:
            embedding_matrix_Mazajak[i] = embedding_vector
    except:
        continue

# Building the model
embedding_vector_length = 300
model_finetune_mazajak = Sequential()
model_finetune_mazajak.add(Embedding(vocab_size, embedding_vector_length, weights=[embedding_matrix_Mazajak]))
model_finetune_mazajak.add(GlobalAveragePooling1D())
model_finetune_mazajak.add(Dropout(0.2))
model_finetune_mazajak.add(Dense(18, activation='softmax'))
model_finetune_mazajak.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

# callback and early stopping
mc = ModelCheckpoint('best_model_finetune_mazajak.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', verbose=1, patience=2, min_delta= .1)

# traing the model
history_finetune_mazajak = model_finetune_mazajak.fit(padded_sequence,y, validation_data=(val_padded_sequence, y_val_categorical),
                                               epochs=10, batch_size=32, callbacks=[es, mc])

# tokenize the test data and padding
test_tweets = tokenizer.texts_to_sequences(X_test['pure_tweet'])
test_padded_sequence = pad_sequences(test_tweets, maxlen=60)

# printint the results on test data
test_pred = model_finetune_mazajak.predict(test_padded_sequence)
print("Deep learning results")
print('accuracy: ', np.mean(list(map(np.argmax,test_pred))==y_test),' ||F1 score: ', f1_score(y_test, np.argmax(test_pred,axis=1), average='macro'))

# Machine learning
#=================
two_gram_svm = Pipeline([
    ('tfidf',TfidfVectorizer(ngram_range=(1,2))),
    ('clf', LinearSVC()),
])
two_gram_svm.fit(X_train['pure_tweet'], y_train)
joblib.dump(two_gram_svm, filename= 'two_gram_svm.joblib')

# print the results on the test data
print("Machine learning results")
print("accuracy: ", two_gram_svm.score(X_test['pure_tweet'], y_test))
print("macro F1 score: ", f1_score(y_test, two_gram_svm.predict(X_test['pure_tweet']), average='macro'))














