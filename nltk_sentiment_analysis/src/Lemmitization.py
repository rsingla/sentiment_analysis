from string import ascii_lowercase

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


base_path = "../../nltk_sentiment_analysis/resources/"
list_play_files = ['julius_caesar',"macbeth", "merchantofvenice","romeoandjuliet", "tempest"]

list_small_play_files = ['julius_caesar']

"""
Future article:
https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python
"""

def parse_play_data():
    word_dict = {}
    for play_name in list_play_files:
        tokens = []

        path = base_path+play_name
        file = open(path,"r")

        for line in file:
            if len(line) > 0:
                tokens = tokens + wordpunct_tokenize(line)
        file.close()

        lemmatized_list = apply_lemmatizer(tokens)
        pre_processed_list = clean_out_punct(lemmatized_list)
        final_list = remove_stop_words(pre_processed_list)

        word_dict[play_name] = final_list

    return word_dict

def clean_out_punct(tokens):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    pre_processed_list = []
    for token in tokens:
        if(len(tokenizer.tokenize(token)) > 0):
            pre_processed_list = pre_processed_list + tokenizer.tokenize(token)
    return pre_processed_list

def apply_stemmer(complete_data):

    stemmer = PorterStemmer()
    stem_list = []
    for word in complete_data:
        stem_list.append(stemmer.stem(word))

    return stem_list

def apply_lemmatizer(complete_data):
    lemmatizer = WordNetLemmatizer()

    lemmatized_list = []
    for word in complete_data:
        lemmatized_list.append(lemmatizer.lemmatize(word))

    return lemmatized_list

def remove_stop_words(complete_data):

    stop_words = set(stopwords.words("english"))

    clean_list = []

    for wrd in complete_data:
        if wrd.lower() not in stop_words and len(wrd) > 1:
            clean_list.append(wrd.lower())

    return clean_list

def printfile(word_dict):
    for key,value in word_dict.items():
        print(key)
        #print(value)
        freq = nltk.FreqDist(value)
        freq.plot(50, cumulative=False, title=key)

word_dict = parse_play_data()
printfile(word_dict)
