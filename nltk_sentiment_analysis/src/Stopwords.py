from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

base_path = "../../nltk_sentiment_analysis/resources/"
list_play_files = ['julius_caesar',"macbeth", "merchantofvenice","romeoandjuliet", "tempest"]


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

        final_list = remove_stop_words(complete_data)
        word_dict[play_name] = final_list

    return word_dict

def remove_stop_words(complete_data):

    stop_words = set(stopwords.words("english"))

    for stopwrd in stop_words:
        if stopwrd in complete_data:
            complete_data.remove(stopwrd)
    return complete_data

def printfile(word_dict):
    for key,value in word_dict.items():
        print(key)
        print(len(value))
        #print(value)

word_dict = parse_play_data()
printfile(word_dict)
