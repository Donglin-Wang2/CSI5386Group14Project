import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.image import resize
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import nltk
from nltk.tokenize import word_tokenize
import math

class VQASequence(Sequence):

    def __init__(self, batch_size, df, img_size=224):
        
        glove_df = pd.read_csv('glove.840B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
        self.glove = {key: val.values for key, val in glove_df.T.items()}
        
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
    
    def __getitem__(self, idx):

        batch_df = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]

        #Embed the question
        max_question_length = 0
   
        questions = batch_df['Question']
        embedded_questions = []
        for q in questions:
            embed_q = []
            for t in word_tokenize(q):
                if t in self.glove.keys():
                    embed_q.append(self.glove[t])
            embedded_questions.append(embed_q)
            max_question_length = max(max_question_length, len(embed_q))
        
        empty_vec = [[0]*300]

        for i in range(len(embedded_questions)):
            while len(embedded_questions[i]) != max_question_length:
                embedded_questions[i] = embedded_questions[i] + (empty_vec)
        
        # print(np.asarray(embedded_questions).shape)
        embedded_questions = np.asarray(embedded_questions).astype('float32')

        #Load images
        img_paths = batch_df['Image']
        images = []
        for p in img_paths:
            img = load_img(p)
            img_arr = img_to_array(img)
            resize_img_arr = resize(img_arr, [self.img_size, self.img_size])
            images.append(resize_img_arr/255)
        images = np.asarray(images).astype('float32')

        #Make the answer binary
        answers = batch_df['Answer']
        map = {
            "yes":1,
            "no":0
        }
        answers_encoded = []
        for i in answers:
            answers_encoded.append(map[i])
        answers_encoded = np.asarray(answers_encoded)

        return (images, embedded_questions), answers_encoded
    
    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)