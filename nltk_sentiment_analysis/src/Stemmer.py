from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import *

base_path = "../../nltk_sentiment_analysis/resources/"
list_play_files = ['julius_caesar',"macbeth", "merchantofvenice","romeoandjuliet", "tempest"]

"""
Future article:
https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python
"""

def parse_play_data():
    word_dict = {}
    for play_name in list_play_files:
        complete_data = []

        path = base_path+play_name
        file = open(path,"r")

        for line in file:
            if len(line) > 0:
                complete_data = complete_data + word_tokenize(line)
        file.close()

        stem_word_list = remove_stop_words(complete_data)
        final_list = apply_stemmer(stem_word_list)

        word_dict[play_name] = final_list

    return word_dict

def apply_stemmer(complete_data):

    stemmer = PorterStemmer()
    stem_list = []
    for word in complete_data:
        stem_list.append(stemmer.stem(word))

    return stem_list

def remove_stop_words(complete_data):

    stop_words = set(stopwords.words("english"))

    for stopwrd in stop_words:
        if stopwrd in complete_data:
            complete_data.remove(stopwrd)

    return complete_data

def printfile(word_dict):
    for key,value in word_dict.items():
        print(key)
        print(value)

word_dict = parse_play_data()
printfile(word_dict)
