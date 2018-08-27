import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize, PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer


base_path = "../../nltk_sentiment_analysis/resources/"

"""
Future article:
https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python
"""

def parse_play_data(play_list):
    word_dict = {}
    for play_name in play_list:
        tokens = []

        path = base_path+play_name
        file = open(path,"r")

        #custom_tokenizer = PunktSentenceTokenizer()

        for line in file:
            if len(line) > 0:
                tokens = tokens + wordpunct_tokenize(line)
        file.close()

        lemmatized_list = apply_lemmatizer(tokens)
        pre_processed_list = clean_out_punct(lemmatized_list)
        stop_wrd_cleaned_list = remove_stop_words(pre_processed_list)

        final_list = create_tagged_list(stop_wrd_cleaned_list)

        word_dict[play_name] = final_list

    return word_dict

def create_tagged_list(tokens):
    list_tagged_wrds = []
    for token in tokens:
        tagged_tokens = word_tokenize(token)
        tagged = nltk.pos_tag(tagged_tokens)
        list_tagged_wrds = list_tagged_wrds + tagged

    return list_tagged_wrds


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
        print(value)
        freq = nltk.FreqDist(value)
        freq.plot(20, cumulative=False, title=key)


list_play_files = ['julius_caesar',"macbeth", "merchantofvenice","romeoandjuliet", "tempest"]
list_small_play_files = ['julius_caesar']

word_dict = parse_play_data(list_play_files)
printfile(word_dict)
