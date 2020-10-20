import os
import re
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

from variables import*

def get_class_names():
    return os.listdir(train_dir)

def load_image_data():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rescale = rescale,
                                    rotation_range = rotation_range,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    width_shift_range=shift_range,
                                    height_shift_range=shift_range,
                                    horizontal_flip = True,
                                    validation_split= val_split
                                    )

    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size_cnn,
                                    classes = get_class_names(),
                                    shuffle = True)

    validation_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size_cnn,
                                    classes = get_class_names(),
                                    shuffle = True)

    return train_generator, validation_generator

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path, encoding='ISO 8859-1')
    df = df[['Discription', 'Breed']]
    df['Breed'] = df['Breed'].str.lower()
    df = df.dropna(axis=1, how='all') # drop columns which  
    df = df[df['Discription'].notna()]
    df = df.fillna(method='ffill')
    return df

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(review):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def preprocessed_data(reviews):
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        for review in reviews:
            updated_review = preprocess_one(review)
            updated_reviews.append(updated_review)
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)
    
def load_text_data():
    df = preprocess_data(csv_path)
    classes = df['Breed'].values 

    encoder = LabelEncoder()
    encoder.fit(classes)
    classes = encoder.transform(classes)
    doggy_reviews = df['Discription'].values
    doggy_reviews = preprocessed_data(doggy_reviews)

    Ntest = int(val_split * len(classes))
    X, Y = shuffle(doggy_reviews, classes)
    Xtrain, Xtest = X[:-Ntest], X[-Ntest:]
    Ytrain, Ytest = Y[:-Ntest], Y[-Ntest:]
    return Xtrain, Xtest, Ytrain, Ytest

def load_pretrained_embeddings(word2index):
    if os.path.exists(embedding_matrix_path):
        print("Pretrained embedding weights loading")
        embedding_matrix = np.load(embedding_matrix_path)
    else:
        embeddings_index = {}
        for line in open(word_embedding_path):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
        open(word_embedding_path).close()

        embedding_matrix = np.zeros((len(word2index) + 1, get_embedding_dim()))
        for word, i in word2index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                if word == oov_tok:
                    embedding_matrix[i] = embeddings_index.get('unk')

        np.save(embedding_matrix_path, embedding_matrix)
        print("Pretrained embedding weights saving")
    return embedding_matrix

def get_embedding_dim():
    glove_text = os.path.split(word_embedding_path)[-1]
    return int(glove_text.split('.')[-2][:-1])

