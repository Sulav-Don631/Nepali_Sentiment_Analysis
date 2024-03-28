from flask import Flask, request, jsonify, render_template
import re
import pickle
import math
from collections import Counter
import numpy as np
from final_code import TFIDFVectorizer

app = Flask(__name__)

stop_words_file = open("/Users/sudinshrestha/Desktop/ultrafinal/NepaliSentimentAnalysis-main/stopwords.txt","r",encoding="utf-8")
stop_words = stop_words_file.read()
stop_words = stop_words.split("\n")

@app.route('/')
def index():
    return render_template('index.html')

def data_cleaning(string):
    text = re.sub('\,|\@|\-|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:|\-|\?|।|/|\—|\०|\१|\२|\३|\४|\५|\६|\७|\८|\९|[0-9]', '', string)
    return text

def stop_word_remove(array_element):
    array_element_set = set(array_element)
    final_list = list(array_element_set.difference(stop_words))
    return final_list

class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary = None
        self.idf = None

    def fit_transform(self, corpus):
        # Build vocabulary
        self.vocabulary = set()
        for document in corpus:
            self.vocabulary.update(document)
        self.vocabulary = list(self.vocabulary)

        # Calculate IDF
        idf = {}
        N = len(corpus)
        for term in self.vocabulary:
            df = sum(1 for document in corpus if term in document)
            idf[term] = math.log(N / (1 + df))

        # Transform documents to TF-IDF representation
        tfidf_matrix = np.zeros((len(corpus), len(self.vocabulary)))
        for i, document in enumerate(corpus):
            tf = Counter(document)
            total_terms = len(document)
            for j, term in enumerate(self.vocabulary):
                if total_terms != 0:
                    tfidf_matrix[i, j] = (tf.get(term, 0) / total_terms) * idf[term]
                else:
                    tfidf_matrix[i, j] = 0  # Set TF-IDF to 0 if total_terms is 0

        self.idf = idf
        return tfidf_matrix

    def transform(self, corpus):
        if corpus is None:
            print("Error")
            return None
        tfidf_matrix = np.zeros((len(corpus), len(self.vocabulary)))
        for i, document in enumerate(corpus):
            tf = Counter(document)
            total_terms = len(document)
            for j, term in enumerate(self.vocabulary):
                if total_terms != 0:
                    tfidf_matrix[i, j] = (tf.get(term, 0) / total_terms) * self.idf.get(term, 0)
                else:
                    tfidf_matrix[i, j] = 0  # Set TF-IDF to 0 if total_terms is 0
        return tfidf_matrix
    

with open('/Users/sudinshrestha/Desktop/ultrafinal/NepaliSentimentAnalysis-main/classify_data.pickle', 'rb') as pickle_saved_data:
    unpickled_data = pickle.load(pickle_saved_data)

@app.route('/process_text', methods=['POST'])
def process_text():
    text = request.form['text']
    print("Received text:", text)
    clean = data_cleaning(text)
    print(clean)
    tokenized_sentence = clean.split()
    print(tokenized_sentence)
    stop_word_removed = stop_word_remove(tokenized_sentence)
    print(stop_word_removed)
    tfidf = TFIDFVectorizer()
    tfidf.fit_transform(stop_word_removed)
    tfidf_word = tfidf.transform(stop_word_removed)
    print(tfidf_word)
    
    return render_template('index.html',received_text=text, 
                           cleaned_text=clean, 
                           tokenized_sentence=tokenized_sentence,
                           stop_word_removed=stop_word_removed)

 


if __name__ == '__main__':
    app.run(debug=True)
